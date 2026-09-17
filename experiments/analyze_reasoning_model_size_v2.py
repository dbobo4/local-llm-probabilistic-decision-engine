from __future__ import annotations

import json
import re
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results" / "binary_reasoning_model_size_v2_calibration.json"


def contains_boolean_label(text: str) -> bool:
    return re.search(r"\b(True|False)\b", text, flags=re.IGNORECASE) is not None


def contains_judgment_word(text: str) -> bool:
    return re.search(
        r"\b(correct|incorrect|true|false|valid|invalid)\b",
        text,
        flags=re.IGNORECASE,
    ) is not None


def summarize_model(model: dict) -> None:
    rows = model["examples"]

    correct = [
        row for row in rows
        if row["reasoning_conditioned_direct"]["correct"]
    ]
    incorrect = [
        row for row in rows
        if not row["reasoning_conditioned_direct"]["correct"]
    ]

    direct_wrong_reasoned_right = [
        row for row in rows
        if not row["direct"]["correct"]
        and row["reasoning_conditioned_direct"]["correct"]
    ]

    direct_right_reasoned_wrong = [
        row for row in rows
        if row["direct"]["correct"]
        and not row["reasoning_conditioned_direct"]["correct"]
    ]

    label_count = sum(
        contains_boolean_label(row["reasoning"]["text"])
        for row in rows
    )

    judgment_count = sum(
        contains_judgment_word(row["reasoning"]["text"])
        for row in rows
    )

    print()
    print("=" * 80)
    print(model["model"])
    print("=" * 80)

    print("Direct accuracy:   ", model["direct_accuracy"])
    print(
        "Reasoned accuracy: ",
        model["reasoning_conditioned_direct_accuracy"],
    )

    print()
    print(
        "Rationale tokens mean:   ",
        statistics.mean(
            row["reasoning"]["generated_output_tokens"]
            for row in rows
        ),
    )
    print(
        "Rationale tokens median: ",
        statistics.median(
            row["reasoning"]["generated_output_tokens"]
            for row in rows
        ),
    )

    if correct:
        print(
            "Mean tokens when reasoned CORRECT:",
            statistics.mean(
                row["reasoning"]["generated_output_tokens"]
                for row in correct
            ),
        )

    if incorrect:
        print(
            "Mean tokens when reasoned WRONG:  ",
            statistics.mean(
                row["reasoning"]["generated_output_tokens"]
                for row in incorrect
            ),
        )

    print()
    print(
        "Rationales containing literal True/False:",
        f"{label_count}/{len(rows)}",
    )
    print(
        "Rationales containing judgment words:    ",
        f"{judgment_count}/{len(rows)}",
    )

    print()
    print(
        "Direct wrong -> reasoned correct:",
        len(direct_wrong_reasoned_right),
    )
    print(
        "Direct correct -> reasoned wrong:",
        len(direct_right_reasoned_wrong),
    )

    print()
    print("--- First 10 fixes ---")

    for row in direct_wrong_reasoned_right[:10]:
        rationale = row["reasoning"]["text"].replace("\n", " ")
        print()
        print(
            row["id"],
            "| target=",
            row["target"],
            "| tokens=",
            row["reasoning"]["generated_output_tokens"],
        )
        print(rationale[:500])

    print()
    print("--- All reasoned errors ---")

    for row in incorrect:
        rationale = row["reasoning"]["text"].replace("\n", " ")
        print()
        print(
            row["id"],
            "| target=",
            row["target"],
            "| direct=",
            row["direct"]["selected"],
            "| reasoned=",
            row["reasoning_conditioned_direct"]["selected"],
            "| tokens=",
            row["reasoning"]["generated_output_tokens"],
        )
        print(rationale[:700])


def main() -> None:
    payload = json.loads(INPUT.read_text(encoding="utf-8"))

    models = payload["models"]

    for model in models:
        summarize_model(model)

    if len(models) == 2:
        first = {
            row["id"]: row
            for row in models[0]["examples"]
        }

        second = {
            row["id"]: row
            for row in models[1]["examples"]
        }

        first_only = []
        second_only = []
        both_wrong = []

        for example_id in first:
            a = first[example_id]
            b = second[example_id]

            a_correct = a["reasoning_conditioned_direct"]["correct"]
            b_correct = b["reasoning_conditioned_direct"]["correct"]

            if a_correct and not b_correct:
                first_only.append(example_id)
            elif b_correct and not a_correct:
                second_only.append(example_id)
            elif not a_correct and not b_correct:
                both_wrong.append(example_id)

        print()
        print("=" * 80)
        print("CROSS-MODEL REASONING COMPARISON")
        print("=" * 80)

        print("1.5B correct / 3B wrong:", len(first_only))
        print("3B correct / 1.5B wrong:", len(second_only))
        print("Both wrong:             ", len(both_wrong))

        print()
        print("1.5B-only correct IDs:")
        for value in first_only:
            print(" ", value)

        print()
        print("3B-only correct IDs:")
        for value in second_only:
            print(" ", value)

        print()
        print("Both wrong IDs:")
        for value in both_wrong:
            print(" ", value)


if __name__ == "__main__":
    main()
