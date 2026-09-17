from __future__ import annotations

import gc
import hashlib
import json
import re
import statistics
import time
from pathlib import Path

import torch

from llm_decision_engine import DecisionEngine


ROOT = Path(__file__).resolve().parents[1]

DATASET = (
    ROOT
    / "benchmarks"
    / "data"
    / "binary_quality_v2_calibration.json"
)

PREVIOUS_RESULTS = (
    ROOT
    / "results"
    / "binary_reasoning_model_size_v2_calibration.json"
)

OUTPUT = (
    ROOT
    / "results"
    / "binary_reasoning_protocol_v3_calibration.json"
)

EXPECTED_DATASET_SHA256 = (
    "d2116a228b48c959f3c77e8e4ecfa816"
    "033aa4044ec55bbfb930553985a2e4b8"
)

MODEL_IDS = (
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
)

MAX_NEW_TOKENS = 96

PROTOCOLS = (
    "freeform_verification",
    "structured_verification",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def make_reasoning_prefix(
    tokenizer,
    state: str,
    question: str,
    protocol: str,
) -> str:
    if protocol == "freeform_verification":
        instruction = (
            "Verify the statement by doing only the arithmetic, "
            "divisibility check, parity check, or comparison needed. "
            "Show the verification work briefly. "
            "Do not give a verdict. "
            "Do not use the words True, False, correct, incorrect, "
            "yes, or no."
        )

    elif protocol == "structured_verification":
        instruction = (
            "Verify the statement without giving a verdict. "
            "Use exactly these three lines:\n"
            "GIVEN: restate the relevant mathematical claim.\n"
            "COMPUTED: independently compute or check the relevant value.\n"
            "REFERENCE: state the value, parity, divisibility condition, "
            "or relation claimed by the original statement.\n"
            "Do not use the words True, False, correct, incorrect, "
            "yes, or no."
        )

    else:
        raise ValueError(f"Unknown protocol: {protocol}")

    content = (
        f"STATE:\n{state}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"{instruction}"
    )

    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )


def generate_rationale(
    engine: DecisionEngine,
    prefix: str,
) -> tuple[str, int, float]:
    encoded = engine.tokenizer(
        prefix,
        return_tensors="pt",
        add_special_tokens=False,
    )

    device = next(engine.model.parameters()).device

    encoded = {
        key: value.to(device)
        for key, value in encoded.items()
    }

    input_length = encoded["input_ids"].shape[1]

    synchronize()
    start = time.perf_counter()

    with torch.inference_mode():
        output = engine.model.generate(
            **encoded,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=engine.tokenizer.eos_token_id,
        )

    synchronize()
    elapsed = time.perf_counter() - start

    generated_ids = output[0, input_length:]

    text = engine.tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()

    return (
        text,
        int(generated_ids.numel()),
        elapsed,
    )


def accuracy(
    selections: list[bool],
    targets: list[bool],
) -> float:
    return sum(
        selection == target
        for selection, target in zip(selections, targets)
    ) / len(targets)


def target_accuracy(
    selections: list[bool],
    targets: list[bool],
    target_value: bool,
) -> float:
    pairs = [
        (selection, target)
        for selection, target in zip(selections, targets)
        if target == target_value
    ]

    return sum(
        selection == target
        for selection, target in pairs
    ) / len(pairs)


def contains_literal_label(text: str) -> bool:
    return re.search(
        r"\b(True|False)\b",
        text,
        flags=re.IGNORECASE,
    ) is not None


def contains_evaluative_word(text: str) -> bool:
    return re.search(
        r"\b(correct|incorrect)\b",
        text,
        flags=re.IGNORECASE,
    ) is not None


def contains_yes_no(text: str) -> bool:
    return re.search(
        r"\b(yes|no)\b",
        text,
        flags=re.IGNORECASE,
    ) is not None


def nonempty_line_count(text: str) -> int:
    return len(
        [
            line
            for line in text.splitlines()
            if line.strip()
        ]
    )


def run_protocol(
    engine: DecisionEngine,
    model_id: str,
    examples: list[dict],
    protocol: str,
    baseline_by_id: dict[str, dict],
) -> dict:
    print()
    print("-" * 80)
    print("PROTOCOL:", protocol)
    print("-" * 80)

    targets = []
    selections = []

    token_counts = []
    latencies = []

    rows = []

    for index, example in enumerate(examples, start=1):
        prefix = make_reasoning_prefix(
            engine.tokenizer,
            example["state"],
            example["question"],
            protocol,
        )

        rationale, tokens, seconds = generate_rationale(
            engine,
            prefix,
        )

        reasoning_state = (
            f"{example['state']}\n\n"
            "VERIFICATION WORK:\n"
            f"{rationale}"
        )

        result = engine.boolean(
            state=reasoning_state,
            question=example["question"],
            scoring="sum",
            execution="sequential",
        )

        target = example["target"]

        targets.append(target)
        selections.append(result.selected)

        token_counts.append(tokens)
        latencies.append(seconds)

        baseline = baseline_by_id[example["id"]]

        rows.append(
            {
                "id": example["id"],
                "category": example["category"],
                "target": target,
                "baseline_direct_selected": (
                    baseline["direct"]["selected"]
                ),
                "baseline_direct_correct": (
                    baseline["direct"]["correct"]
                ),
                "rationale": rationale,
                "rationale_tokens": tokens,
                "rationale_seconds": seconds,
                "hit_token_limit": (
                    tokens >= MAX_NEW_TOKENS
                ),
                "literal_true_false": (
                    contains_literal_label(rationale)
                ),
                "evaluative_correct_incorrect": (
                    contains_evaluative_word(rationale)
                ),
                "contains_yes_no": (
                    contains_yes_no(rationale)
                ),
                "nonempty_line_count": (
                    nonempty_line_count(rationale)
                ),
                "selected": result.selected,
                "probability_true": result.probability_true,
                "score_delta": (
                    result.scores["yes"]
                    - result.scores["no"]
                ),
                "correct": result.selected == target,
            }
        )

        print(
            f"{index:3d}/{len(examples)}  "
            f'{example["id"]:<40}  '
            f"target={str(target):5}  "
            f"selected={str(result.selected):5}  "
            f"tokens={tokens:3d}"
        )

    overall_accuracy = accuracy(
        selections,
        targets,
    )

    true_accuracy = target_accuracy(
        selections,
        targets,
        True,
    )

    false_accuracy = target_accuracy(
        selections,
        targets,
        False,
    )

    fixes = sum(
        not row["baseline_direct_correct"]
        and row["correct"]
        for row in rows
    )

    breaks = sum(
        row["baseline_direct_correct"]
        and not row["correct"]
        for row in rows
    )

    categories = sorted(
        {
            example["category"]
            for example in examples
        }
    )

    category_results = []

    for category in categories:
        category_rows = [
            row
            for row in rows
            if row["category"] == category
        ]

        category_accuracy = sum(
            row["correct"]
            for row in category_rows
        ) / len(category_rows)

        category_results.append(
            {
                "category": category,
                "accuracy": category_accuracy,
            }
        )

    result = {
        "model": model_id,
        "protocol": protocol,
        "accuracy": overall_accuracy,
        "true_accuracy": true_accuracy,
        "false_accuracy": false_accuracy,
        "fixes_baseline_direct_errors": fixes,
        "breaks_baseline_direct_correct": breaks,
        "rationale_token_mean": statistics.mean(
            token_counts
        ),
        "rationale_token_median": statistics.median(
            token_counts
        ),
        "rationale_latency_median_ms": (
            statistics.median(latencies) * 1000.0
        ),
        "hit_token_limit": sum(
            tokens >= MAX_NEW_TOKENS
            for tokens in token_counts
        ),
        "literal_true_false_count": sum(
            row["literal_true_false"]
            for row in rows
        ),
        "evaluative_correct_incorrect_count": sum(
            row["evaluative_correct_incorrect"]
            for row in rows
        ),
        "yes_no_count": sum(
            row["contains_yes_no"]
            for row in rows
        ),
        "exactly_three_nonempty_lines": sum(
            row["nonempty_line_count"] == 3
            for row in rows
        ),
        "categories": category_results,
        "examples": rows,
    }

    print()
    print("Accuracy:       ", f"{overall_accuracy:.6f}")
    print("True accuracy:  ", f"{true_accuracy:.6f}")
    print("False accuracy: ", f"{false_accuracy:.6f}")

    print()
    print("Fixes:  ", fixes)
    print("Breaks: ", breaks)

    print()
    print(
        "Token mean:     ",
        f"{statistics.mean(token_counts):.2f}",
    )
    print(
        "Token median:   ",
        f"{statistics.median(token_counts):.2f}",
    )
    print(
        "Median latency: ",
        f"{statistics.median(latencies) * 1000.0:.2f} ms",
    )

    print()
    print(
        "Contains True/False:       ",
        f"{result['literal_true_false_count']}/120",
    )
    print(
        "Contains correct/incorrect:",
        f"{result['evaluative_correct_incorrect_count']}/120",
    )
    print(
        "Contains yes/no:           ",
        f"{result['yes_no_count']}/120",
    )

    if protocol == "structured_verification":
        print(
            "Exactly 3 nonempty lines: ",
            f"{result['exactly_three_nonempty_lines']}/120",
        )

    print()
    print("By category:")

    for row in category_results:
        print(
            f'  {row["category"]:<16}'
            f'{row["accuracy"]:.3f}'
        )

    return result


def main() -> None:
    dataset_hash = sha256_file(DATASET)

    if dataset_hash != EXPECTED_DATASET_SHA256:
        raise RuntimeError(
            "Calibration dataset hash mismatch."
        )

    examples = json.loads(
        DATASET.read_text(encoding="utf-8")
    )

    previous = json.loads(
        PREVIOUS_RESULTS.read_text(encoding="utf-8")
    )

    previous_models = {
        model["model"]: model
        for model in previous["models"]
    }

    all_results = []

    for model_id in MODEL_IDS:
        print()
        print("=" * 80)
        print("MODEL:", model_id)
        print("=" * 80)

        previous_model = previous_models[model_id]

        print(
            "Previous direct accuracy:   ",
            previous_model["direct_accuracy"],
        )

        print(
            "Previous reasoned accuracy: ",
            previous_model[
                "reasoning_conditioned_direct_accuracy"
            ],
        )

        baseline_by_id = {
            row["id"]: row
            for row in previous_model["examples"]
        }

        engine = DecisionEngine(
            model=model_id
        )

        protocol_results = []

        for protocol in PROTOCOLS:
            protocol_results.append(
                run_protocol(
                    engine,
                    model_id,
                    examples,
                    protocol,
                    baseline_by_id,
                )
            )

        all_results.append(
            {
                "model": model_id,
                "previous_direct_accuracy": (
                    previous_model["direct_accuracy"]
                ),
                "previous_reasoned_accuracy": (
                    previous_model[
                        "reasoning_conditioned_direct_accuracy"
                    ]
                ),
                "protocols": protocol_results,
            }
        )

        del engine
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    payload = {
        "split": "calibration",
        "dataset": str(
            DATASET.relative_to(ROOT)
        ),
        "dataset_sha256": dataset_hash,
        "test_split_used": False,
        "max_new_tokens": MAX_NEW_TOKENS,
        "models": all_results,
    }

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("FINAL PROTOCOL COMPARISON")
    print("=" * 80)

    print(
        f'{"Model":<32}'
        f'{"Direct":>9}'
        f'{"Old reason":>12}'
        f'{"Freeform":>11}'
        f'{"Structured":>12}'
    )

    for model in all_results:
        protocols = {
            row["protocol"]: row
            for row in model["protocols"]
        }

        print(
            f'{model["model"]:<32}'
            f'{model["previous_direct_accuracy"]:>9.3f}'
            f'{model["previous_reasoned_accuracy"]:>12.3f}'
            f'{protocols["freeform_verification"]["accuracy"]:>11.3f}'
            f'{protocols["structured_verification"]["accuracy"]:>12.3f}'
        )

    print()
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()
