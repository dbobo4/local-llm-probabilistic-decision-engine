from __future__ import annotations

import gc
import hashlib
import json
import math
import statistics
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

from llm_decision_engine import DecisionEngine


ROOT = Path(__file__).resolve().parents[1]

DATASET = (
    ROOT
    / "benchmarks"
    / "data"
    / "routing_v1_calibration.json"
)

OUTPUT = (
    ROOT
    / "results"
    / "routing_v1_calibration_direct.json"
)

EXPECTED_DATASET_SHA256 = (
    "7f94ca16628c3591a69eb38342ee0386"
    "35b0ef72f04f045262fc6145646759c0"
)

MODEL_IDS = (
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
)

ROUTING_QUESTION = "Which support category best matches this customer message?"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def evaluate_metrics(
    rows: list[dict],
    candidates: list[str],
) -> dict:
    accuracy = statistics.mean(
        1.0 if row["selected"] == row["target"] else 0.0
        for row in rows
    )

    top2_accuracy = statistics.mean(
        1.0 if row["target"] in row["top2"] else 0.0
        for row in rows
    )

    eps = 1e-12

    nll = statistics.mean(
        -math.log(
            max(
                row["probabilities"][row["target"]],
                eps,
            )
        )
        for row in rows
    )

    brier_values = []

    for row in rows:
        value = 0.0

        for candidate in candidates:
            target_value = (
                1.0
                if candidate == row["target"]
                else 0.0
            )

            value += (
                row["probabilities"][candidate]
                - target_value
            ) ** 2

        brier_values.append(value)

    brier = statistics.mean(
        brier_values
    )

    mean_confidence = statistics.mean(
        row["confidence"]
        for row in rows
    )

    ece = 0.0
    num_bins = 10

    for bin_index in range(num_bins):
        lower = bin_index / num_bins
        upper = (bin_index + 1) / num_bins

        members = [
            row
            for row in rows
            if (
                row["confidence"] >= lower
                and (
                    row["confidence"] < upper
                    or (
                        bin_index == num_bins - 1
                        and row["confidence"] <= upper
                    )
                )
            )
        ]

        if not members:
            continue

        bin_confidence = statistics.mean(
            row["confidence"]
            for row in members
        )

        bin_accuracy = statistics.mean(
            1.0
            if row["selected"] == row["target"]
            else 0.0
            for row in members
        )

        ece += (
            len(members)
            / len(rows)
            * abs(
                bin_confidence
                - bin_accuracy
            )
        )

    return {
        "accuracy": accuracy,
        "top2_accuracy": top2_accuracy,
        "multiclass_brier_sum": brier,
        "nll": nll,
        "ece_top_label": ece,
        "mean_top1_confidence": mean_confidence,
    }


def per_class_accuracy(
    rows: list[dict],
    candidates: list[str],
) -> dict[str, float]:
    output = {}

    for candidate in candidates:
        subset = [
            row
            for row in rows
            if row["target"] == candidate
        ]

        output[candidate] = statistics.mean(
            1.0
            if row["selected"] == row["target"]
            else 0.0
            for row in subset
        )

    return output


def confusion_matrix(
    rows: list[dict],
    candidates: list[str],
) -> dict:
    matrix = {
        target: {
            predicted: 0
            for predicted in candidates
        }
        for target in candidates
    }

    for row in rows:
        matrix[
            row["target"]
        ][
            row["selected"]
        ] += 1

    return matrix


def main() -> None:
    dataset_hash = sha256_file(
        DATASET
    )

    if dataset_hash != EXPECTED_DATASET_SHA256:
        raise RuntimeError(
            "Calibration dataset SHA256 mismatch."
        )

    payload = json.loads(
        DATASET.read_text(
            encoding="utf-8"
        )
    )

    if payload["split"] != "calibration":
        raise RuntimeError(
            "Expected routing calibration split."
        )

    candidates = list(
        payload["candidates"]
    )

    examples = payload[
        "examples"
    ]

    if len(examples) != 120:
        raise RuntimeError(
            f"Expected 120 examples, got {len(examples)}."
        )

    print("ROUTING V1 DIRECT CALIBRATION")
    print("=============================")
    print()
    print("Dataset:", DATASET.relative_to(ROOT))
    print("SHA256: ", dataset_hash)
    print("Commit: ", git_head())
    print("Examples:", len(examples))
    print("Classes: ", len(candidates))
    print("Chance accuracy:", f"{1 / len(candidates):.6f}")
    print()
    print("Candidates:")
    for candidate in candidates:
        print(" ", candidate)

    all_models = []

    for model_id in MODEL_IDS:
        print()
        print("=" * 80)
        print("MODEL:", model_id)
        print("=" * 80)

        engine = DecisionEngine(
            model=model_id
        )

        rows = []
        total_generated_tokens = 0
        observed_token_counts = None

        for index, example in enumerate(
            examples,
            start=1,
        ):
            started = time.perf_counter()

            result = engine.choice(
                state=example["text"],
                question=ROUTING_QUESTION,
                candidates=candidates,
                scoring="sum",
                execution="sequential",
            )

            elapsed_ms = (
                time.perf_counter() - started
            ) * 1000.0

            probabilities = {
                candidate: float(
                    result.probabilities[candidate]
                )
                for candidate in candidates
            }

            scores = {
                candidate: float(
                    result.scores[candidate]
                )
                for candidate in candidates
            }

            probability_sum = sum(
                probabilities.values()
            )

            if abs(
                probability_sum - 1.0
            ) > 1e-5:
                raise RuntimeError(
                    "Candidate probabilities do not sum to 1."
                )

            ranked = sorted(
                candidates,
                key=lambda candidate: (
                    probabilities[candidate]
                ),
                reverse=True,
            )

            selected = str(
                result.selected
            )

            if selected not in candidates:
                raise RuntimeError(
                    f"Unexpected selected candidate: {selected!r}"
                )

            generated_tokens = int(
                result.generated_output_tokens
            )

            total_generated_tokens += (
                generated_tokens
            )

            if generated_tokens != 0:
                raise RuntimeError(
                    "Direct routing generated output tokens."
                )

            token_counts = {
                key: int(value)
                for key, value
                in result.token_counts.items()
            }

            if observed_token_counts is None:
                observed_token_counts = (
                    token_counts
                )

            row = {
                "id": example["id"],
                "family": example["family"],
                "text": example["text"],
                "target": example["intent"],
                "selected": selected,
                "correct": (
                    selected
                    == example["intent"]
                ),
                "top2": ranked[:2],
                "confidence": (
                    probabilities[selected]
                ),
                "target_probability": (
                    probabilities[
                        example["intent"]
                    ]
                ),
                "probabilities": probabilities,
                "scores": scores,
                "token_counts": token_counts,
                "generated_output_tokens": (
                    generated_tokens
                ),
                "latency_ms": elapsed_ms,
            }

            rows.append(row)

            print(
                f"{index:3d}/120  "
                f'{example["id"]:<52} '
                f'target={example["intent"]:<18} '
                f'pred={selected:<18} '
                f'p={probabilities[selected]:.3f} '
                f'{"OK" if row["correct"] else "ERR"}'
            )

        metrics = evaluate_metrics(
            rows,
            candidates,
        )

        class_accuracy = (
            per_class_accuracy(
                rows,
                candidates,
            )
        )

        confusion = confusion_matrix(
            rows,
            candidates,
        )

        latencies = [
            row["latency_ms"]
            for row in rows
        ]

        predicted_counts = Counter(
            row["selected"]
            for row in rows
        )

        print()
        print("--- SUMMARY ---")
        print(
            "Accuracy:             ",
            f'{metrics["accuracy"]:.6f}',
        )
        print(
            "Top-2 accuracy:       ",
            f'{metrics["top2_accuracy"]:.6f}',
        )
        print(
            "Multiclass Brier sum: ",
            f'{metrics["multiclass_brier_sum"]:.6f}',
        )
        print(
            "NLL:                  ",
            f'{metrics["nll"]:.6f}',
        )
        print(
            "ECE (top-label):      ",
            f'{metrics["ece_top_label"]:.6f}',
        )
        print(
            "Mean confidence:      ",
            f'{metrics["mean_top1_confidence"]:.6f}',
        )
        print(
            "Latency median:       ",
            f"{statistics.median(latencies):.2f} ms",
        )
        print(
            "Latency mean:         ",
            f"{statistics.mean(latencies):.2f} ms",
        )
        print(
            "Generated tokens:     ",
            total_generated_tokens,
        )

        print()
        print("--- CANDIDATE TOKEN COUNTS ---")

        for candidate in candidates:
            print(
                f"{candidate:<20}"
                f"{observed_token_counts[candidate]}"
            )

        print()
        print("--- PER-CLASS ACCURACY ---")

        for candidate in candidates:
            print(
                f"{candidate:<20}"
                f"{class_accuracy[candidate]:.3f}"
            )

        print()
        print("--- PREDICTED CLASS COUNTS ---")

        for candidate in candidates:
            print(
                f"{candidate:<20}"
                f"{predicted_counts[candidate]}"
            )

        print()
        print("--- CONFUSION MATRIX ---")
        print(
            f'{"target \\ predicted":<22}'
            + "".join(
                f"{i:>6}"
                for i in range(
                    len(candidates)
                )
            )
        )

        for target in candidates:
            print(
                f"{target:<22}"
                + "".join(
                    f"{confusion[target][predicted]:>6}"
                    for predicted in candidates
                )
            )

        print()
        print("Column mapping:")

        for index, candidate in enumerate(
            candidates
        ):
            print(
                f"  {index}: {candidate}"
            )

        all_models.append(
            {
                "model": model_id,
                "scoring": "sum",
                "execution": "sequential",
                "generated_output_tokens": (
                    total_generated_tokens
                ),
                "candidate_token_counts": (
                    observed_token_counts
                ),
                "metrics": metrics,
                "latency": {
                    "mean_ms": statistics.mean(
                        latencies
                    ),
                    "median_ms": statistics.median(
                        latencies
                    ),
                },
                "per_class_accuracy": (
                    class_accuracy
                ),
                "predicted_class_counts": (
                    dict(predicted_counts)
                ),
                "confusion_matrix": (
                    confusion
                ),
                "examples": rows,
            }
        )

        del engine
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output_payload = {
        "benchmark": "routing_v1",
        "evaluation_type": (
            "calibration_only_direct_routing"
        ),
        "split": "calibration",
        "test_split_used": False,
        "dataset": str(
            DATASET.relative_to(ROOT)
        ),
        "dataset_sha256": dataset_hash,
        "git_commit": git_head(),
        "state_source": "customer message text",
        "question": ROUTING_QUESTION,
        "candidates": candidates,
        "scoring": "sum",
        "execution": "sequential",
        "generated_output_tokens": 0,
        "models": all_models,
    }

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT.write_text(
        json.dumps(
            output_payload,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("FINAL ROUTING CALIBRATION")
    print("=" * 80)

    print(
        f'{"Model":<34}'
        f'{"Accuracy":>10}'
        f'{"Top-2":>10}'
        f'{"NLL":>10}'
        f'{"Brier":>10}'
        f'{"ECE":>10}'
        f'{"Tokens":>10}'
    )

    for model in all_models:
        metrics = model[
            "metrics"
        ]

        print(
            f'{model["model"]:<34}'
            f'{metrics["accuracy"]:>10.3f}'
            f'{metrics["top2_accuracy"]:>10.3f}'
            f'{metrics["nll"]:>10.3f}'
            f'{metrics["multiclass_brier_sum"]:>10.3f}'
            f'{metrics["ece_top_label"]:>10.3f}'
            f'{model["generated_output_tokens"]:>10}'
        )

    print()
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()
