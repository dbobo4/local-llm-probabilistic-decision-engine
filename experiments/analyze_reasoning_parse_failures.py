from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT = (
    ROOT
    / "results"
    / "binary_reasoning_diagnostic_v2_calibration.json"
)


def current_parse(text: str) -> bool | None:
    matches = re.findall(
        r"FINAL\s*:\s*(True|False)\b",
        text,
        flags=re.IGNORECASE,
    )

    if not matches:
        return None

    return matches[-1].lower() == "true"


def relaxed_parse(text: str) -> bool | None:
    patterns = (
        r"FINAL\s*(?:ANSWER)?\s*[:\-]\s*\**(True|False)\b",
        r"FINAL\s*(?:ANSWER)?\s+IS\s+\**(True|False)\b",
        r"(?:FINAL\s+ANSWER|ANSWER)\s*[:\-]\s*\**(True|False)\b",
    )

    for pattern in patterns:
        matches = re.findall(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if matches:
            return matches[-1].lower() == "true"

    return None


def last_boolean_word(text: str) -> bool | None:
    matches = re.findall(
        r"\b(True|False)\b",
        text,
        flags=re.IGNORECASE,
    )

    if not matches:
        return None

    return matches[-1].lower() == "true"


def accuracy(rows, key: str) -> tuple[int, int, float]:
    parsed = [
        row
        for row in rows
        if row[key] is not None
    ]

    correct = sum(
        row[key] == row["target"]
        for row in parsed
    )

    return (
        correct,
        len(parsed),
        correct / len(parsed) if parsed else 0.0,
    )


def main() -> None:
    payload = json.loads(
        INPUT.read_text(encoding="utf-8")
    )

    rows = []

    for example in payload["examples"]:
        text = example["reasoning"]["text"]
        target = example["target"]

        rows.append(
            {
                "id": example["id"],
                "target": target,
                "direct": example["direct"]["selected"],
                "tokens": example["reasoning"][
                    "generated_output_tokens"
                ],
                "text": text,
                "current": current_parse(text),
                "relaxed": relaxed_parse(text),
                "last_boolean": last_boolean_word(text),
            }
        )

    failures = [
        row
        for row in rows
        if row["current"] is None
    ]

    print("=== CURRENT PARSER ===")
    print("Total examples:       ", len(rows))
    print("Parsed:               ", len(rows) - len(failures))
    print("Unparsed:             ", len(failures))
    print(
        "Hit 128-token limit:  ",
        sum(row["tokens"] >= 128 for row in failures),
    )
    print(
        "Contains 'FINAL':     ",
        sum(
            "final" in row["text"].lower()
            for row in failures
        ),
    )
    print(
        "Contains True/False:  ",
        sum(
            re.search(
                r"\b(True|False)\b",
                row["text"],
                flags=re.IGNORECASE,
            )
            is not None
            for row in failures
        ),
    )

    print()
    print("Unparsed token-count distribution:")
    counts = Counter(
        row["tokens"]
        for row in failures
    )

    for tokens, count in counts.most_common(15):
        print(f"  {tokens:3d} tokens : {count}")

    for key, label in (
        ("current", "Current exact FINAL parser"),
        ("relaxed", "Relaxed FINAL parser"),
        ("last_boolean", "Last True/False word diagnostic"),
    ):
        correct, parsed, acc = accuracy(rows, key)

        print()
        print(label)
        print(f"  parsed:   {parsed}/120")
        print(f"  correct:  {correct}/{parsed}")
        print(f"  accuracy: {acc:.6f}")

        if parsed:
            comparable = [
                row
                for row in rows
                if row[key] is not None
            ]

            direct_correct = sum(
                row["direct"] == row["target"]
                for row in comparable
            )

            print(
                "  direct accuracy on SAME subset: "
                f"{direct_correct}/{parsed} "
                f"({direct_correct / parsed:.6f})"
            )

    print()
    print("=== FIRST 15 CURRENT PARSE FAILURES ===")

    for row in failures[:15]:
        tail = row["text"][-500:].replace("\n", "\\n")

        print()
        print(
            f'{row["id"]} | '
            f'target={row["target"]} | '
            f'tokens={row["tokens"]}'
        )
        print(tail)


if __name__ == "__main__":
    main()
