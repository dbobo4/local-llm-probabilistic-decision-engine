from __future__ import annotations

import json
from pathlib import Path

from llm_decision_engine import DecisionEngine


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "benchmarks" / "data" / "binary_quality_v1.json"


def main() -> None:
    examples = json.loads(DATASET.read_text(encoding="utf-8"))
    engine = DecisionEngine(model="Qwen/Qwen2.5-1.5B-Instruct")

    forward_correct = 0
    reverse_correct = 0
    changed_selection = 0
    deltas = []

    print(
        f"{'id':>12}  {'target':>6}  "
        f"{'Pyes y/n':>10}  {'Pyes n/y':>10}  "
        f"{'delta':>10}  {'sel y/n':>7}  {'sel n/y':>7}"
    )

    for example in examples:
        forward = engine.choice(
            state=example["state"],
            question=example["question"],
            candidates=["yes", "no"],
            execution="sequential",
        )

        reverse = engine.choice(
            state=example["state"],
            question=example["question"],
            candidates=["no", "yes"],
            execution="sequential",
        )

        p_yes_forward = forward.probabilities["yes"]
        p_yes_reverse = reverse.probabilities["yes"]
        delta = p_yes_reverse - p_yes_forward
        deltas.append(abs(delta))

        selected_forward = forward.selected == "yes"
        selected_reverse = reverse.selected == "yes"
        target = example["target"]

        forward_correct += selected_forward == target
        reverse_correct += selected_reverse == target
        changed_selection += selected_forward != selected_reverse

        print(
            f'{example["id"]:>12}  '
            f"{str(target):>6}  "
            f"{p_yes_forward:10.6f}  "
            f"{p_yes_reverse:10.6f}  "
            f"{delta:+10.6f}  "
            f"{str(selected_forward):>7}  "
            f"{str(selected_reverse):>7}"
        )

    count = len(examples)

    print()
    print("--- Order invariance summary ---")
    print(f"Examples:              {count}")
    print(f"Accuracy yes/no:       {forward_correct / count:.6f}")
    print(f"Accuracy no/yes:       {reverse_correct / count:.6f}")
    print(f"Selections changed:    {changed_selection}/{count}")
    print(f"Mean absolute delta:   {sum(deltas) / count:.6f}")
    print(f"Max absolute delta:    {max(deltas):.6f}")


if __name__ == "__main__":
    main()
