from __future__ import annotations

import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "benchmarks" / "data"

QUESTION = "Is the statement correct?"
CATEGORIES = (
    "addition",
    "subtraction",
    "multiplication",
    "parity",
    "divisibility",
    "comparison",
)

EXAMPLES_PER_CATEGORY_PER_LABEL = 10


def nonzero_offset(rng: random.Random, magnitude: int = 7) -> int:
    values = [value for value in range(-magnitude, magnitude + 1) if value != 0]
    return rng.choice(values)


def addition_example(rng: random.Random, target: bool) -> str:
    a = rng.randint(2, 99)
    b = rng.randint(2, 99)
    correct = a + b
    shown = correct if target else correct + nonzero_offset(rng)
    return f"{a} + {b} = {shown}."


def subtraction_example(rng: random.Random, target: bool) -> str:
    a = rng.randint(20, 150)
    b = rng.randint(2, a - 1)
    correct = a - b
    shown = correct if target else correct + nonzero_offset(rng)
    return f"{a} - {b} = {shown}."


def multiplication_example(rng: random.Random, target: bool) -> str:
    a = rng.randint(2, 19)
    b = rng.randint(2, 19)
    correct = a * b
    shown = correct if target else correct + nonzero_offset(rng, 12)
    return f"{a} times {b} equals {shown}."


def parity_example(rng: random.Random, target: bool) -> str:
    number = rng.randint(10, 199)
    actually_even = number % 2 == 0
    claimed_even = actually_even if target else not actually_even
    adjective = "even" if claimed_even else "odd"
    return f"{number} is an {adjective} number."


def divisibility_example(rng: random.Random, target: bool) -> str:
    divisor = rng.randint(2, 12)

    if target:
        multiplier = rng.randint(2, 20)
        number = divisor * multiplier
    else:
        multiplier = rng.randint(2, 20)
        remainder = rng.randint(1, divisor - 1)
        number = divisor * multiplier + remainder

    return f"{number} is divisible by {divisor}."


def comparison_example(rng: random.Random, target: bool) -> str:
    a = rng.randint(1, 200)
    b = rng.randint(1, 200)

    while a == b:
        b = rng.randint(1, 200)

    true_relation = "greater than" if a > b else "less than"

    if target:
        relation = true_relation
    else:
        relation = "less than" if true_relation == "greater than" else "greater than"

    return f"{a} is {relation} {b}."


GENERATORS = {
    "addition": addition_example,
    "subtraction": subtraction_example,
    "multiplication": multiplication_example,
    "parity": parity_example,
    "divisibility": divisibility_example,
    "comparison": comparison_example,
}


def build_split(
    name: str,
    *,
    seed: int,
    seen_states: set[str],
) -> list[dict]:
    rng = random.Random(seed)
    examples = []

    for category in CATEGORIES:
        generator = GENERATORS[category]

        for target in (True, False):
            created = 0

            while created < EXAMPLES_PER_CATEGORY_PER_LABEL:
                state = generator(rng, target)

                if state in seen_states:
                    continue

                seen_states.add(state)

                examples.append(
                    {
                        "id": f"{name}_{category}_{str(target).lower()}_{created + 1:02d}",
                        "category": category,
                        "state": state,
                        "question": QUESTION,
                        "target": target,
                    }
                )

                created += 1

    rng.shuffle(examples)
    return examples


def validate_split(examples: list[dict], expected_count: int) -> None:
    assert len(examples) == expected_count
    assert sum(example["target"] for example in examples) == expected_count // 2
    assert sum(not example["target"] for example in examples) == expected_count // 2
    assert len({example["state"] for example in examples}) == expected_count

    for category in CATEGORIES:
        category_examples = [
            example for example in examples
            if example["category"] == category
        ]

        assert len(category_examples) == 20
        assert sum(example["target"] for example in category_examples) == 10
        assert sum(not example["target"] for example in category_examples) == 10


def write_json(path: Path, payload: list[dict]) -> None:
    path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    seen_states: set[str] = set()

    calibration = build_split(
        "calibration",
        seed=20260917,
        seen_states=seen_states,
    )

    test = build_split(
        "test",
        seed=20260918,
        seen_states=seen_states,
    )

    validate_split(calibration, 120)
    validate_split(test, 120)

    calibration_states = {example["state"] for example in calibration}
    test_states = {example["state"] for example in test}

    assert calibration_states.isdisjoint(test_states)

    calibration_path = OUTPUT_DIR / "binary_quality_v2_calibration.json"
    test_path = OUTPUT_DIR / "binary_quality_v2_test.json"

    write_json(calibration_path, calibration)
    write_json(test_path, test)

    print("Calibration examples:", len(calibration))
    print("  True:", sum(example["target"] for example in calibration))
    print("  False:", sum(not example["target"] for example in calibration))
    print("Test examples:", len(test))
    print("  True:", sum(example["target"] for example in test))
    print("  False:", sum(not example["target"] for example in test))
    print("Cross-split state overlap:", len(calibration_states & test_states))

    print()
    print("Per-category counts:")

    for category in CATEGORIES:
        calibration_count = sum(
            example["category"] == category
            for example in calibration
        )
        test_count = sum(
            example["category"] == category
            for example in test
        )

        print(
            f"{category:>14}: "
            f"calibration={calibration_count:3d}, "
            f"test={test_count:3d}"
        )

    print()
    print("Saved:", calibration_path)
    print("Saved:", test_path)


if __name__ == "__main__":
    main()
