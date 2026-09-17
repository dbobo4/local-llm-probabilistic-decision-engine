from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from llm_decision_engine import (
    DecisionEngine,
    evaluate_binary_calibration,
    evaluate_binary_predictions,
)


ROOT = Path(__file__).resolve().parents[1]

DATASET = (
    ROOT
    / "benchmarks"
    / "data"
    / "binary_quality_v2_calibration.json"
)

OUTPUT = (
    ROOT
    / "results"
    / "binary_quality_v2_calibration_scores.json"
)

EXPECTED_DATASET_SHA256 = (
    "d2116a228b48c959f3c77e8e4ecfa816"
    "033aa4044ec55bbfb930553985a2e4b8"
)

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def main() -> None:
    dataset_hash = sha256_file(DATASET)

    if dataset_hash != EXPECTED_DATASET_SHA256:
        raise RuntimeError(
            "Calibration dataset hash does not match the frozen v2 split. "
            f"Expected {EXPECTED_DATASET_SHA256}, got {dataset_hash}."
        )

    examples = json.loads(DATASET.read_text(encoding="utf-8"))

    if len(examples) != 120:
        raise RuntimeError(
            f"Expected 120 calibration examples, got {len(examples)}."
        )

    engine = DecisionEngine(model=MODEL_ID)

    predictions = []
    probabilities_true = []
    targets = []

    print(f"Model:       {MODEL_ID}")
    print(f"Dataset:     {DATASET.relative_to(ROOT)}")
    print(f"SHA256:      {dataset_hash}")
    print(f"Examples:    {len(examples)}")
    print("Execution:   sequential")
    print("Scoring:     sum")
    print()

    for index, example in enumerate(examples, start=1):
        target = example["target"]

        if not isinstance(target, bool):
            raise TypeError(
                f'Example {example["id"]!r} target must be boolean.'
            )

        result = engine.boolean(
            state=example["state"],
            question=example["question"],
            scoring="sum",
            execution="sequential",
        )

        score_true = result.scores["yes"]
        score_false = result.scores["no"]
        score_delta = score_true - score_false

        probabilities_true.append(result.probability_true)
        targets.append(target)

        predictions.append(
            {
                "id": example["id"],
                "category": example["category"],
                "state": example["state"],
                "question": example["question"],
                "target": target,
                "score_true": score_true,
                "score_false": score_false,
                "score_delta": score_delta,
                "raw_probability_true": result.probability_true,
                "raw_probability_false": result.probability_false,
                "selected": result.selected,
                "correct": result.selected == target,
                "generated_output_tokens": result.generated_output_tokens,
            }
        )

        print(
            f"{index:3d}/{len(examples)}  "
            f'{example["id"]:<40}  '
            f"target={str(target):5}  "
            f"delta={score_delta:+9.4f}  "
            f"p_true={result.probability_true:.6f}  "
            f"correct={result.selected == target}"
        )

    evaluation = evaluate_binary_predictions(
        probabilities_true,
        targets,
    )

    calibration = evaluate_binary_calibration(
        probabilities_true,
        targets,
        num_bins=10,
    )

    total_generated_tokens = sum(
        prediction["generated_output_tokens"]
        for prediction in predictions
    )

    payload = {
        "model": MODEL_ID,
        "dataset": str(DATASET.relative_to(ROOT)),
        "dataset_sha256": dataset_hash,
        "split": "calibration",
        "execution": "sequential",
        "scoring": "sum",
        "example_count": len(examples),
        "generated_output_tokens": total_generated_tokens,
        "raw_evaluation": asdict(evaluation),
        "raw_calibration": asdict(calibration),
        "predictions": predictions,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print("--- Raw calibration-split metrics ---")
    print(f"Accuracy:   {evaluation.accuracy:.6f}")
    print(f"Mean Brier: {evaluation.mean_brier:.6f}")
    print(f"Mean NLL:   {evaluation.mean_nll:.6f}")
    print(
        "ECE:        "
        f"{calibration.expected_calibration_error:.6f}"
    )
    print(f"Generated output tokens: {total_generated_tokens}")
    print()
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()
