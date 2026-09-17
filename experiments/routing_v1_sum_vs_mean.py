from __future__ import annotations

import gc
import hashlib
import json
import math
import statistics
import time
from collections import Counter
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

SUM_RESULTS = (
    ROOT
    / "results"
    / "routing_v1_calibration_direct.json"
)

OUTPUT = (
    ROOT
    / "results"
    / "routing_v1_sum_vs_mean_calibration.json"
)

EXPECTED_DATASET_SHA256 = (
    "7f94ca16628c3591a69eb38342ee0386"
    "35b0ef72f04f045262fc6145646759c0"
)

MODEL_IDS = (
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
)

ROUTING_QUESTION = (
    "Which support category best matches this customer message?"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def metrics(
    rows: list[dict],
    candidates: list[str],
) -> dict:
    accuracy = statistics.mean(
        1.0 if row["selected"] == row["target"] else 0.0
        for row in rows
    )

    top2 = statistics.mean(
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

    return {
        "accuracy": accuracy,
        "top2_accuracy": top2,
        "nll": nll,
        "multiclass_brier_sum": statistics.mean(
            brier_values
        ),
        "mean_confidence": statistics.mean(
            row["probabilities"][row["selected"]]
            for row in rows
        ),
    }


def per_class(
    rows: list[dict],
    candidates: list[str],
) -> dict:
    result = {}

    for candidate in candidates:
        subset = [
            row
            for row in rows
            if row["target"] == candidate
        ]

        result[candidate] = statistics.mean(
            1.0 if row["selected"] == row["target"] else 0.0
            for row in subset
        )

    return result


def main() -> None:
    dataset_hash = sha256_file(
        DATASET
    )

    if dataset_hash != EXPECTED_DATASET_SHA256:
        raise RuntimeError(
            "Calibration dataset SHA256 mismatch."
        )

    dataset_payload = json.loads(
        DATASET.read_text(
            encoding="utf-8"
        )
    )

    candidates = list(
        dataset_payload["candidates"]
    )

    examples = dataset_payload[
        "examples"
    ]

    sum_payload = json.loads(
        SUM_RESULTS.read_text(
            encoding="utf-8"
        )
    )

    if sum_payload["dataset_sha256"] != EXPECTED_DATASET_SHA256:
        raise RuntimeError(
            "Stored sum baseline dataset SHA mismatch."
        )

    sum_models = {
        model["model"]: model
        for model in sum_payload["models"]
    }

    output_models = []

    for model_id in MODEL_IDS:
        print()
        print("=" * 80)
        print("MODEL:", model_id)
        print("=" * 80)

        sum_model = sum_models[
            model_id
        ]

        sum_rows_by_id = {
            row["id"]: row
            for row in sum_model["examples"]
        }

        engine = DecisionEngine(
            model=model_id
        )

        mean_rows = []

        for index, example in enumerate(
            examples,
            start=1,
        ):
            started = time.perf_counter()

            result = engine.choice(
                state=example["text"],
                question=ROUTING_QUESTION,
                candidates=candidates,
                scoring="mean",
                execution="sequential",
            )

            latency_ms = (
                time.perf_counter()
                - started
            ) * 1000.0

            probabilities = {
                candidate: float(
                    result.probabilities[candidate]
                )
                for candidate in candidates
            }

            ranked = sorted(
                candidates,
                key=lambda candidate: probabilities[candidate],
                reverse=True,
            )

            selected = str(
                result.selected
            )

            if result.generated_output_tokens != 0:
                raise RuntimeError(
                    "Mean scoring generated output tokens."
                )

            mean_rows.append(
                {
                    "id": example["id"],
                    "target": example["intent"],
                    "family": example["family"],
                    "text": example["text"],
                    "selected": selected,
                    "correct": (
                        selected
                        == example["intent"]
                    ),
                    "top2": ranked[:2],
                    "probabilities": probabilities,
                    "scores": {
                        candidate: float(
                            result.scores[candidate]
                        )
                        for candidate in candidates
                    },
                    "token_counts": {
                        key: int(value)
                        for key, value
                        in result.token_counts.items()
                    },
                    "latency_ms": latency_ms,
                    "generated_output_tokens": 0,
                }
            )

            sum_selected = str(
                sum_rows_by_id[
                    example["id"]
                ]["selected"]
            )

            marker = (
                "SAME"
                if sum_selected == selected
                else "SWITCH"
            )

            print(
                f"{index:3d}/120  "
                f'{example["id"]:<52} '
                f'target={example["intent"]:<18} '
                f'sum={sum_selected:<18} '
                f'mean={selected:<18} '
                f'{marker}'
            )

        mean_metrics = metrics(
            mean_rows,
            candidates,
        )

        sum_rows = [
            sum_rows_by_id[
                example["id"]
            ]
            for example in examples
        ]

        sum_metrics = metrics(
            sum_rows,
            candidates,
        )

        sum_class = per_class(
            sum_rows,
            candidates,
        )

        mean_class = per_class(
            mean_rows,
            candidates,
        )

        switches = []

        fixed = 0
        broken = 0
        neutral_switch = 0

        for sum_row, mean_row in zip(
            sum_rows,
            mean_rows,
        ):
            if (
                sum_row["selected"]
                == mean_row["selected"]
            ):
                continue

            sum_correct = (
                sum_row["selected"]
                == sum_row["target"]
            )

            mean_correct = (
                mean_row["selected"]
                == mean_row["target"]
            )

            if (
                not sum_correct
                and mean_correct
            ):
                outcome = "fixed"
                fixed += 1

            elif (
                sum_correct
                and not mean_correct
            ):
                outcome = "broken"
                broken += 1

            else:
                outcome = "neutral"
                neutral_switch += 1

            switches.append(
                {
                    "id": mean_row["id"],
                    "target": mean_row["target"],
                    "sum_selected": (
                        sum_row["selected"]
                    ),
                    "mean_selected": (
                        mean_row["selected"]
                    ),
                    "outcome": outcome,
                }
            )

        predicted_sum = Counter(
            row["selected"]
            for row in sum_rows
        )

        predicted_mean = Counter(
            row["selected"]
            for row in mean_rows
        )

        print()
        print("--- GLOBAL RESULTS ---")
        print(
            f'{"Method":<12}'
            f'{"Accuracy":>10}'
            f'{"Top-2":>10}'
            f'{"NLL":>10}'
            f'{"Brier":>10}'
            f'{"Conf":>10}'
        )

        for name, result in (
            ("sum", sum_metrics),
            ("mean", mean_metrics),
        ):
            print(
                f'{name:<12}'
                f'{result["accuracy"]:>10.3f}'
                f'{result["top2_accuracy"]:>10.3f}'
                f'{result["nll"]:>10.3f}'
                f'{result["multiclass_brier_sum"]:>10.3f}'
                f'{result["mean_confidence"]:>10.3f}'
            )

        print()
        print("--- DECISION SWITCHES ---")
        print(
            "Changed decisions:",
            len(switches),
        )
        print(
            "Fixed errors:      ",
            fixed,
        )
        print(
            "Broken correct:    ",
            broken,
        )
        print(
            "Neutral switches:  ",
            neutral_switch,
        )
        print(
            "Net fixes:         ",
            fixed - broken,
        )

        print()
        print("--- PER-CLASS ACCURACY ---")
        print(
            f'{"Class":<20}'
            f'{"Tokens":>8}'
            f'{"Sum":>10}'
            f'{"Mean":>10}'
            f'{"Delta":>10}'
        )

        token_counts = (
            mean_rows[0]["token_counts"]
        )

        for candidate in candidates:
            delta = (
                mean_class[candidate]
                - sum_class[candidate]
            )

            print(
                f'{candidate:<20}'
                f'{token_counts[candidate]:>8}'
                f'{sum_class[candidate]:>10.3f}'
                f'{mean_class[candidate]:>10.3f}'
                f'{delta:>+10.3f}'
            )

        print()
        print("--- PREDICTED CLASS COUNTS ---")
        print(
            f'{"Class":<20}'
            f'{"Tokens":>8}'
            f'{"Sum":>10}'
            f'{"Mean":>10}'
        )

        for candidate in candidates:
            print(
                f'{candidate:<20}'
                f'{token_counts[candidate]:>8}'
                f'{predicted_sum[candidate]:>10}'
                f'{predicted_mean[candidate]:>10}'
            )

        latencies = [
            row["latency_ms"]
            for row in mean_rows
        ]

        print()
        print(
            "Mean-scoring median latency:",
            f"{statistics.median(latencies):.2f} ms",
        )

        output_models.append(
            {
                "model": model_id,
                "sum": {
                    "metrics": sum_metrics,
                    "per_class_accuracy": (
                        sum_class
                    ),
                    "predicted_counts": (
                        dict(predicted_sum)
                    ),
                },
                "mean": {
                    "metrics": mean_metrics,
                    "per_class_accuracy": (
                        mean_class
                    ),
                    "predicted_counts": (
                        dict(predicted_mean)
                    ),
                    "median_latency_ms": (
                        statistics.median(
                            latencies
                        )
                    ),
                    "examples": mean_rows,
                },
                "decision_switches": {
                    "count": len(switches),
                    "fixed": fixed,
                    "broken": broken,
                    "neutral": neutral_switch,
                    "net_fixes": (
                        fixed - broken
                    ),
                    "examples": switches,
                },
                "candidate_token_counts": (
                    token_counts
                ),
            }
        )

        del engine
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    OUTPUT.write_text(
        json.dumps(
            {
                "benchmark": "routing_v1",
                "evaluation_type": (
                    "calibration_only_sum_vs_mean"
                ),
                "test_split_used": False,
                "dataset_sha256": (
                    dataset_hash
                ),
                "question": (
                    ROUTING_QUESTION
                ),
                "generated_output_tokens": 0,
                "models": output_models,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("FINAL SUM VS MEAN")
    print("=" * 80)

    print(
        f'{"Model":<34}'
        f'{"Sum acc":>10}'
        f'{"Mean acc":>10}'
        f'{"Delta":>10}'
        f'{"Switches":>10}'
        f'{"Net fixes":>11}'
    )

    for model in output_models:
        sum_acc = (
            model["sum"]["metrics"]["accuracy"]
        )

        mean_acc = (
            model["mean"]["metrics"]["accuracy"]
        )

        print(
            f'{model["model"]:<34}'
            f'{sum_acc:>10.3f}'
            f'{mean_acc:>10.3f}'
            f'{mean_acc - sum_acc:>+10.3f}'
            f'{model["decision_switches"]["count"]:>10}'
            f'{model["decision_switches"]["net_fixes"]:>11}'
        )

    print()
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()
