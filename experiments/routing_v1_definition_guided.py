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

BASELINE_RESULTS = (
    ROOT
    / "results"
    / "routing_v1_calibration_direct.json"
)

OUTPUT = (
    ROOT
    / "results"
    / "routing_v1_definition_guided_calibration.json"
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


def compute_metrics(
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

    return {
        "accuracy": accuracy,
        "top2_accuracy": top2_accuracy,
        "nll": nll,
        "multiclass_brier_sum": statistics.mean(
            brier_values
        ),
        "mean_confidence": statistics.mean(
            row["probabilities"][row["selected"]]
            for row in rows
        ),
    }


def compute_per_class(
    rows: list[dict],
    candidates: list[str],
) -> dict[str, float]:
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


def build_guided_state(
    text: str,
    candidates: list[str],
    definitions: dict[str, str],
) -> str:
    policy = "\n".join(
        f"{candidate}: {definitions[candidate]}"
        for candidate in candidates
    )

    return (
        "ROUTING POLICY:\n"
        f"{policy}\n\n"
        "CUSTOMER MESSAGE:\n"
        f"{text}"
    )


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

    if dataset_payload["split"] != "calibration":
        raise RuntimeError(
            "Expected calibration split."
        )

    candidates = list(
        dataset_payload["candidates"]
    )

    definitions = dict(
        dataset_payload["intent_definitions"]
    )

    examples = dataset_payload[
        "examples"
    ]

    if len(examples) != 120:
        raise RuntimeError(
            f"Expected 120 examples, got {len(examples)}."
        )

    if set(definitions) != set(candidates):
        raise RuntimeError(
            "Intent definitions do not match candidates."
        )

    baseline_payload = json.loads(
        BASELINE_RESULTS.read_text(
            encoding="utf-8"
        )
    )

    if (
        baseline_payload["dataset_sha256"]
        != EXPECTED_DATASET_SHA256
    ):
        raise RuntimeError(
            "Stored baseline dataset SHA mismatch."
        )

    baseline_models = {
        model["model"]: model
        for model in baseline_payload["models"]
    }

    print("ROUTING V1 DEFINITION-GUIDED CALIBRATION")
    print("========================================")
    print()
    print("Dataset SHA256:", dataset_hash)
    print("Examples:", len(examples))
    print("Test split used: False")
    print()
    print("--- FROZEN INTENT DEFINITIONS ---")

    for candidate in candidates:
        print(
            f"{candidate}: "
            f"{definitions[candidate]}"
        )

    output_models = []

    for model_id in MODEL_IDS:
        print()
        print("=" * 80)
        print("MODEL:", model_id)
        print("=" * 80)

        baseline_model = baseline_models[
            model_id
        ]

        baseline_rows_by_id = {
            row["id"]: row
            for row in baseline_model["examples"]
        }

        engine = DecisionEngine(
            model=model_id
        )

        guided_rows = []

        for index, example in enumerate(
            examples,
            start=1,
        ):
            state = build_guided_state(
                example["text"],
                candidates,
                definitions,
            )

            started = time.perf_counter()

            result = engine.choice(
                state=state,
                question=ROUTING_QUESTION,
                candidates=candidates,
                scoring="sum",
                execution="sequential",
            )

            latency_ms = (
                time.perf_counter()
                - started
            ) * 1000.0

            if result.generated_output_tokens != 0:
                raise RuntimeError(
                    "Definition-guided routing generated output tokens."
                )

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

            ranked = sorted(
                candidates,
                key=lambda candidate: probabilities[candidate],
                reverse=True,
            )

            selected = str(
                result.selected
            )

            baseline_selected = str(
                baseline_rows_by_id[
                    example["id"]
                ]["selected"]
            )

            marker = (
                "SAME"
                if baseline_selected == selected
                else "SWITCH"
            )

            guided_rows.append(
                {
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
                    "probabilities": probabilities,
                    "scores": scores,
                    "token_counts": {
                        key: int(value)
                        for key, value
                        in result.token_counts.items()
                    },
                    "generated_output_tokens": 0,
                    "latency_ms": latency_ms,
                }
            )

            print(
                f"{index:3d}/120  "
                f'{example["id"]:<52} '
                f'target={example["intent"]:<18} '
                f'bare={baseline_selected:<18} '
                f'guided={selected:<18} '
                f'{marker}'
            )

        baseline_rows = [
            baseline_rows_by_id[
                example["id"]
            ]
            for example in examples
        ]

        baseline_metrics = compute_metrics(
            baseline_rows,
            candidates,
        )

        guided_metrics = compute_metrics(
            guided_rows,
            candidates,
        )

        baseline_class = compute_per_class(
            baseline_rows,
            candidates,
        )

        guided_class = compute_per_class(
            guided_rows,
            candidates,
        )

        switches = []
        fixed = 0
        broken = 0
        neutral = 0

        for baseline_row, guided_row in zip(
            baseline_rows,
            guided_rows,
        ):
            if (
                baseline_row["selected"]
                == guided_row["selected"]
            ):
                continue

            baseline_correct = (
                baseline_row["selected"]
                == baseline_row["target"]
            )

            guided_correct = (
                guided_row["selected"]
                == guided_row["target"]
            )

            if (
                not baseline_correct
                and guided_correct
            ):
                outcome = "fixed"
                fixed += 1

            elif (
                baseline_correct
                and not guided_correct
            ):
                outcome = "broken"
                broken += 1

            else:
                outcome = "neutral"
                neutral += 1

            switches.append(
                {
                    "id": guided_row["id"],
                    "target": guided_row["target"],
                    "bare_selected": (
                        baseline_row["selected"]
                    ),
                    "guided_selected": (
                        guided_row["selected"]
                    ),
                    "outcome": outcome,
                }
            )

        predicted_bare = Counter(
            row["selected"]
            for row in baseline_rows
        )

        predicted_guided = Counter(
            row["selected"]
            for row in guided_rows
        )

        print()
        print("--- GLOBAL RESULTS ---")
        print(
            f'{"Method":<14}'
            f'{"Accuracy":>10}'
            f'{"Top-2":>10}'
            f'{"NLL":>10}'
            f'{"Brier":>10}'
            f'{"Conf":>10}'
        )

        for name, result in (
            ("bare", baseline_metrics),
            ("guided", guided_metrics),
        ):
            print(
                f'{name:<14}'
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
            neutral,
        )
        print(
            "Net fixes:         ",
            fixed - broken,
        )

        print()
        print("--- PER-CLASS ACCURACY ---")
        print(
            f'{"Class":<20}'
            f'{"Bare":>10}'
            f'{"Guided":>10}'
            f'{"Delta":>10}'
        )

        for candidate in candidates:
            delta = (
                guided_class[candidate]
                - baseline_class[candidate]
            )

            print(
                f'{candidate:<20}'
                f'{baseline_class[candidate]:>10.3f}'
                f'{guided_class[candidate]:>10.3f}'
                f'{delta:>+10.3f}'
            )

        print()
        print("--- PREDICTED CLASS COUNTS ---")
        print(
            f'{"Class":<20}'
            f'{"Bare":>10}'
            f'{"Guided":>10}'
        )

        for candidate in candidates:
            print(
                f'{candidate:<20}'
                f'{predicted_bare[candidate]:>10}'
                f'{predicted_guided[candidate]:>10}'
            )

        latencies = [
            row["latency_ms"]
            for row in guided_rows
        ]

        print()
        print(
            "Guided median latency:",
            f"{statistics.median(latencies):.2f} ms",
        )
        print(
            "Generated tokens:",
            sum(
                row["generated_output_tokens"]
                for row in guided_rows
            ),
        )

        output_models.append(
            {
                "model": model_id,
                "bare": {
                    "metrics": baseline_metrics,
                    "per_class_accuracy": (
                        baseline_class
                    ),
                    "predicted_counts": (
                        dict(predicted_bare)
                    ),
                },
                "definition_guided": {
                    "metrics": guided_metrics,
                    "per_class_accuracy": (
                        guided_class
                    ),
                    "predicted_counts": (
                        dict(predicted_guided)
                    ),
                    "median_latency_ms": (
                        statistics.median(
                            latencies
                        )
                    ),
                    "generated_output_tokens": 0,
                    "examples": guided_rows,
                },
                "decision_switches": {
                    "count": len(switches),
                    "fixed": fixed,
                    "broken": broken,
                    "neutral": neutral,
                    "net_fixes": (
                        fixed - broken
                    ),
                    "examples": switches,
                },
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
                    "calibration_only_bare_vs_definition_guided"
                ),
                "split": "calibration",
                "test_split_used": False,
                "dataset_sha256": dataset_hash,
                "question": ROUTING_QUESTION,
                "scoring": "sum",
                "execution": "sequential",
                "intent_definitions": definitions,
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
    print("FINAL BARE VS DEFINITION-GUIDED")
    print("=" * 80)

    print(
        f'{"Model":<34}'
        f'{"Bare":>10}'
        f'{"Guided":>10}'
        f'{"Delta":>10}'
        f'{"Switches":>10}'
        f'{"Net fixes":>11}'
    )

    for model in output_models:
        bare_acc = (
            model["bare"]["metrics"]["accuracy"]
        )

        guided_acc = (
            model[
                "definition_guided"
            ]["metrics"]["accuracy"]
        )

        print(
            f'{model["model"]:<34}'
            f'{bare_acc:>10.3f}'
            f'{guided_acc:>10.3f}'
            f'{guided_acc - bare_acc:>+10.3f}'
            f'{model["decision_switches"]["count"]:>10}'
            f'{model["decision_switches"]["net_fixes"]:>11}'
        )

    print()
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()
