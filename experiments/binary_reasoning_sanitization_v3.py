from __future__ import annotations

import gc
import json
import re
from pathlib import Path

import torch

from llm_decision_engine import DecisionEngine


ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT
    / "results"
    / "binary_reasoning_protocol_v3_calibration.json"
)

OUTPUT = (
    ROOT
    / "results"
    / "binary_reasoning_sanitization_v3_calibration.json"
)

EXPECTED_DATASET_SHA256 = (
    "d2116a228b48c959f3c77e8e4ecfa816"
    "033aa4044ec55bbfb930553985a2e4b8"
)


def sanitize_rationale(text: str) -> str:
    sanitized = text

    # Collapse common positive/negative verdict phrases to the SAME neutral marker.
    sanitized = re.sub(
        r"\b(is|was|seems|appears)\s+"
        r"(?:not\s+)?"
        r"(true|false|correct|incorrect)\b",
        r"\1 <verdict>",
        sanitized,
        flags=re.IGNORECASE,
    )

    # Remove any remaining explicit True/False/correct/incorrect polarity.
    sanitized = re.sub(
        r"\b(true|false|correct|incorrect)\b",
        "<verdict>",
        sanitized,
        flags=re.IGNORECASE,
    )

    # Mask explicit answer-style Yes/No, but preserve constructions
    # such as "no remainder".
    sanitized = re.sub(
        r"(?im)^(\s*)(yes|no)\b\s*[:,.-]?\s*",
        r"\1<verdict> ",
        sanitized,
    )

    sanitized = re.sub(
        r"(?i)\b(therefore|thus|so),?\s+(yes|no)\b",
        r"\1, <verdict>",
        sanitized,
    )

    return sanitized


def remaining_explicit_verdict(text: str) -> bool:
    return re.search(
        r"\b(true|false|correct|incorrect)\b",
        text,
        flags=re.IGNORECASE,
    ) is not None


def accuracy(rows: list[dict], field: str) -> float:
    return sum(
        row[field] == row["target"]
        for row in rows
    ) / len(rows)


def target_accuracy(
    rows: list[dict],
    field: str,
    target_value: bool,
) -> float:
    subset = [
        row
        for row in rows
        if row["target"] == target_value
    ]

    return sum(
        row[field] == row["target"]
        for row in subset
    ) / len(subset)


def run_protocol(
    engine: DecisionEngine,
    protocol: dict,
) -> dict:
    rows = []

    for index, example in enumerate(
        protocol["examples"],
        start=1,
    ):
        original_rationale = example["rationale"]

        sanitized_rationale = sanitize_rationale(
            original_rationale
        )

        reasoning_state = (
            f"{example['state'] if 'state' in example else ''}"
        )

        # v3 protocol rows do not store state/question directly,
        # so recover them from the frozen calibration dataset later.
        rows.append(
            {
                "id": example["id"],
                "category": example["category"],
                "target": example["target"],
                "original_selected": example["selected"],
                "original_correct": example["correct"],
                "original_rationale": original_rationale,
                "sanitized_rationale": sanitized_rationale,
                "remaining_explicit_verdict": (
                    remaining_explicit_verdict(
                        sanitized_rationale
                    )
                ),
            }
        )

    return {
        "protocol": protocol["protocol"],
        "rows": rows,
    }


def main() -> None:
    payload = json.loads(
        INPUT.read_text(encoding="utf-8")
    )

    if payload["dataset_sha256"] != EXPECTED_DATASET_SHA256:
        raise RuntimeError(
            "Input results do not match the frozen calibration split."
        )

    dataset_path = (
        ROOT
        / payload["dataset"]
    )

    dataset = json.loads(
        dataset_path.read_text(encoding="utf-8")
    )

    dataset_by_id = {
        example["id"]: example
        for example in dataset
    }

    all_results = []

    for model in payload["models"]:
        model_id = model["model"]

        print()
        print("=" * 80)
        print("MODEL:", model_id)
        print("=" * 80)

        engine = DecisionEngine(
            model=model_id
        )

        model_results = []

        for protocol in model["protocols"]:
            print()
            print("-" * 80)
            print("PROTOCOL:", protocol["protocol"])
            print("-" * 80)

            rows = []

            for index, old_row in enumerate(
                protocol["examples"],
                start=1,
            ):
                example = dataset_by_id[
                    old_row["id"]
                ]

                sanitized = sanitize_rationale(
                    old_row["rationale"]
                )

                reasoning_state = (
                    f"{example['state']}\n\n"
                    "VERIFICATION WORK:\n"
                    f"{sanitized}"
                )

                result = engine.boolean(
                    state=reasoning_state,
                    question=example["question"],
                    scoring="sum",
                    execution="sequential",
                )

                row = {
                    "id": old_row["id"],
                    "category": old_row["category"],
                    "target": old_row["target"],
                    "original_selected": old_row["selected"],
                    "original_correct": old_row["correct"],
                    "sanitized_selected": result.selected,
                    "sanitized_correct": (
                        result.selected
                        == old_row["target"]
                    ),
                    "original_rationale": (
                        old_row["rationale"]
                    ),
                    "sanitized_rationale": sanitized,
                    "remaining_explicit_verdict": (
                        remaining_explicit_verdict(
                            sanitized
                        )
                    ),
                    "probability_true": (
                        result.probability_true
                    ),
                    "score_delta": (
                        result.scores["yes"]
                        - result.scores["no"]
                    ),
                }

                rows.append(row)

                print(
                    f"{index:3d}/120  "
                    f'{old_row["id"]:<40}  '
                    f"target={str(old_row['target']):5}  "
                    f"old={str(old_row['selected']):5}  "
                    f"sanitized={str(result.selected):5}"
                )

            original_accuracy = sum(
                row["original_correct"]
                for row in rows
            ) / len(rows)

            sanitized_accuracy = accuracy(
                rows,
                "sanitized_selected",
            )

            changed = [
                row
                for row in rows
                if row["original_selected"]
                != row["sanitized_selected"]
            ]

            fixes = sum(
                not row["original_correct"]
                and row["sanitized_correct"]
                for row in rows
            )

            breaks = sum(
                row["original_correct"]
                and not row["sanitized_correct"]
                for row in rows
            )

            categories = []

            for category in sorted(
                {row["category"] for row in rows}
            ):
                subset = [
                    row
                    for row in rows
                    if row["category"] == category
                ]

                original_category = sum(
                    row["original_correct"]
                    for row in subset
                ) / len(subset)

                sanitized_category = sum(
                    row["sanitized_correct"]
                    for row in subset
                ) / len(subset)

                categories.append(
                    {
                        "category": category,
                        "original_accuracy": (
                            original_category
                        ),
                        "sanitized_accuracy": (
                            sanitized_category
                        ),
                    }
                )

            result_summary = {
                "protocol": protocol["protocol"],
                "original_accuracy": original_accuracy,
                "sanitized_accuracy": sanitized_accuracy,
                "accuracy_delta": (
                    sanitized_accuracy
                    - original_accuracy
                ),
                "sanitized_true_accuracy": (
                    target_accuracy(
                        rows,
                        "sanitized_selected",
                        True,
                    )
                ),
                "sanitized_false_accuracy": (
                    target_accuracy(
                        rows,
                        "sanitized_selected",
                        False,
                    )
                ),
                "changed_decisions": len(changed),
                "sanitization_fixes": fixes,
                "sanitization_breaks": breaks,
                "remaining_explicit_verdicts": sum(
                    row["remaining_explicit_verdict"]
                    for row in rows
                ),
                "categories": categories,
                "examples": rows,
            }

            model_results.append(
                result_summary
            )

            print()
            print(
                "Original accuracy:  ",
                f"{original_accuracy:.6f}",
            )
            print(
                "Sanitized accuracy: ",
                f"{sanitized_accuracy:.6f}",
            )
            print(
                "Delta:              ",
                f"{sanitized_accuracy - original_accuracy:+.6f}",
            )
            print(
                "True accuracy:      ",
                f"{result_summary['sanitized_true_accuracy']:.6f}",
            )
            print(
                "False accuracy:     ",
                f"{result_summary['sanitized_false_accuracy']:.6f}",
            )
            print(
                "Changed decisions:  ",
                len(changed),
            )
            print(
                "Sanitization fixes: ",
                fixes,
            )
            print(
                "Sanitization breaks:",
                breaks,
            )
            print(
                "Explicit verdicts remaining:",
                result_summary[
                    "remaining_explicit_verdicts"
                ],
            )

            print()
            print("By category:")

            for category in categories:
                print(
                    f'  {category["category"]:<16}'
                    f'{category["original_accuracy"]:.3f}'
                    f' -> '
                    f'{category["sanitized_accuracy"]:.3f}'
                )

        all_results.append(
            {
                "model": model_id,
                "protocols": model_results,
            }
        )

        del engine
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output_payload = {
        "split": "calibration",
        "test_split_used": False,
        "source_results": str(
            INPUT.relative_to(ROOT)
        ),
        "dataset_sha256": (
            EXPECTED_DATASET_SHA256
        ),
        "sanitization": (
            "Collapse explicit verdict polarity "
            "to a common <verdict> marker."
        ),
        "models": all_results,
    }

    OUTPUT.write_text(
        json.dumps(
            output_payload,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("FINAL SANITIZATION COMPARISON")
    print("=" * 80)

    print(
        f'{"Model":<32}'
        f'{"Protocol":<25}'
        f'{"Original":>10}'
        f'{"Sanitized":>12}'
        f'{"Delta":>10}'
    )

    for model in all_results:
        for protocol in model["protocols"]:
            print(
                f'{model["model"]:<32}'
                f'{protocol["protocol"]:<25}'
                f'{protocol["original_accuracy"]:>10.3f}'
                f'{protocol["sanitized_accuracy"]:>12.3f}'
                f'{protocol["accuracy_delta"]:>+10.3f}'
            )

    print()
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()
