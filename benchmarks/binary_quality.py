from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from llm_decision_engine import DecisionEngine
from llm_decision_engine.evaluation import (
    evaluate_binary_calibration,
    evaluate_binary_predictions,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "benchmarks" / "data" / "binary_quality_v1.json"
DEFAULT_OUTPUT = ROOT / "results" / "binary_quality_v1.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument(
        "--execution",
        choices=["sequential", "batch"],
        default="sequential",
    )
    parser.add_argument(
        "--scoring",
        choices=["sum", "mean"],
        default="sum",
    )
    parser.add_argument("--num-bins", type=int, default=5)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    examples = json.loads(args.dataset.read_text(encoding="utf-8"))

    if not examples:
        raise ValueError("Benchmark dataset must not be empty.")

    targets = []
    probabilities_true = []
    predictions = []

    engine = DecisionEngine(model=args.model)

    print(f"Model:     {args.model}")
    print(f"Execution: {args.execution}")
    print(f"Scoring:   {args.scoring}")
    print(f"Examples:  {len(examples)}")
    print()

    for example in examples:
        target = example["target"]

        if not isinstance(target, bool):
            raise TypeError(
                f'Example {example["id"]!r} target must be boolean.'
            )

        result = engine.boolean(
            state=example["state"],
            question=example["question"],
            scoring=args.scoring,
            execution=args.execution,
        )

        probability_true = result.probability_true

        targets.append(target)
        probabilities_true.append(probability_true)

        predictions.append(
            {
                "id": example["id"],
                "category": example["category"],
                "state": example["state"],
                "question": example["question"],
                "target": target,
                "probability_true": probability_true,
                "probability_false": result.probability_false,
                "selected": result.selected,
                "correct": result.selected == target,
                "generated_output_tokens": result.generated_output_tokens,
            }
        )

        print(
            f'{example["id"]:>12}  '
            f"target={str(target):5}  "
            f"p_true={probability_true:.6f}  "
            f"selected={str(result.selected):5}  "
            f"correct={result.selected == target}"
        )

    evaluation = evaluate_binary_predictions(
        probabilities_true,
        targets,
    )
    calibration = evaluate_binary_calibration(
        probabilities_true,
        targets,
        num_bins=args.num_bins,
    )

    total_generated_tokens = sum(
        prediction["generated_output_tokens"]
        for prediction in predictions
    )

    payload = {
        "model": args.model,
        "dataset": str(args.dataset.relative_to(ROOT)),
        "execution": args.execution,
        "scoring": args.scoring,
        "example_count": len(examples),
        "generated_output_tokens": total_generated_tokens,
        "evaluation": asdict(evaluation),
        "calibration": asdict(calibration),
        "predictions": predictions,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print("--- Summary ---")
    print(f"Accuracy:   {evaluation.accuracy:.6f}")
    print(f"Mean Brier: {evaluation.mean_brier:.6f}")
    print(f"Mean NLL:   {evaluation.mean_nll:.6f}")
    print(
        "ECE:        "
        f"{calibration.expected_calibration_error:.6f}"
    )
    print(f"Generated output tokens: {total_generated_tokens}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
