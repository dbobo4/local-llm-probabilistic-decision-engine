from __future__ import annotations

import json
import re
import statistics
import time
from collections import defaultdict
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

DIRECT_RESULTS = (
    ROOT
    / "results"
    / "binary_quality_v2_calibration_scores.json"
)

OUTPUT = (
    ROOT
    / "results"
    / "binary_reasoning_diagnostic_v2_calibration.json"
)

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

EXPECTED_DATASET_SHA256 = (
    "d2116a228b48c959f3c77e8e4ecfa816"
    "033aa4044ec55bbfb930553985a2e4b8"
)


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def make_direct_prefix(tokenizer, state: str, question: str) -> str:
    content = (
        f"STATE:\n{state}\n\n"
        f"QUESTION:\n{question}\n\n"
        "Answer exactly one word: True or False."
    )

    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )


def make_reasoning_prefix(tokenizer, state: str, question: str) -> str:
    content = (
        f"STATE:\n{state}\n\n"
        f"QUESTION:\n{question}\n\n"
        "Reason briefly and carefully before deciding. "
        "End your response with exactly one final line in this form:\n"
        "FINAL: True\n"
        "or\n"
        "FINAL: False"
    )

    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )


def generate(
    engine: DecisionEngine,
    prefix: str,
    *,
    max_new_tokens: int,
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
            max_new_tokens=max_new_tokens,
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

    return text, int(generated_ids.numel()), elapsed


def parse_direct_generation(text: str) -> bool | None:
    match = re.match(
        r"^\s*(True|False)\b",
        text,
        flags=re.IGNORECASE,
    )

    if match is None:
        return None

    return match.group(1).lower() == "true"


def parse_reasoning_generation(text: str) -> bool | None:
    matches = re.findall(
        r"FINAL\s*:\s*(True|False)\b",
        text,
        flags=re.IGNORECASE,
    )

    if not matches:
        return None

    return matches[-1].lower() == "true"


def strict_accuracy(
    selections: list[bool | None],
    targets: list[bool],
) -> float:
    return sum(
        selection is not None and selection == target
        for selection, target in zip(selections, targets)
    ) / len(targets)


def agreement(
    first: list[bool | None],
    second: list[bool | None],
) -> tuple[int, int, float]:
    comparable = [
        (a, b)
        for a, b in zip(first, second)
        if a is not None and b is not None
    ]

    if not comparable:
        return 0, 0, 0.0

    matches = sum(a == b for a, b in comparable)

    return matches, len(comparable), matches / len(comparable)


def main() -> None:
    examples = json.loads(
        DATASET.read_text(encoding="utf-8")
    )

    direct_payload = json.loads(
        DIRECT_RESULTS.read_text(encoding="utf-8")
    )

    if direct_payload["dataset_sha256"] != EXPECTED_DATASET_SHA256:
        raise RuntimeError(
            "Direct score dump does not match the frozen "
            "calibration dataset."
        )

    if len(examples) != 120:
        raise RuntimeError(
            f"Expected 120 examples, got {len(examples)}."
        )

    direct_by_id = {
        prediction["id"]: prediction
        for prediction in direct_payload["predictions"]
    }

    if set(direct_by_id) != {
        example["id"]
        for example in examples
    }:
        raise RuntimeError(
            "Direct results and calibration dataset IDs differ."
        )

    engine = DecisionEngine(model=MODEL_ID)

    rows = []

    targets = []
    direct_selected = []
    greedy_selected = []
    reasoning_selected = []

    greedy_token_counts = []
    reasoning_token_counts = []

    greedy_times = []
    reasoning_times = []

    print(f"Model:    {MODEL_ID}")
    print(f"Examples: {len(examples)}")
    print()
    print("Comparing:")
    print("  A. direct True/False scoring")
    print("  B. greedy generation with identical binary prompt")
    print("  C. short reasoning generation + FINAL answer")
    print()

    for index, example in enumerate(examples, start=1):
        direct = direct_by_id[example["id"]]

        target = example["target"]
        direct_choice = direct["selected"]

        direct_prefix = make_direct_prefix(
            engine.tokenizer,
            example["state"],
            example["question"],
        )

        greedy_text, greedy_tokens, greedy_seconds = generate(
            engine,
            direct_prefix,
            max_new_tokens=4,
        )

        greedy_choice = parse_direct_generation(
            greedy_text
        )

        reasoning_prefix = make_reasoning_prefix(
            engine.tokenizer,
            example["state"],
            example["question"],
        )

        reasoning_text, reasoning_tokens, reasoning_seconds = generate(
            engine,
            reasoning_prefix,
            max_new_tokens=128,
        )

        reasoning_choice = parse_reasoning_generation(
            reasoning_text
        )

        targets.append(target)
        direct_selected.append(direct_choice)
        greedy_selected.append(greedy_choice)
        reasoning_selected.append(reasoning_choice)

        greedy_token_counts.append(greedy_tokens)
        reasoning_token_counts.append(reasoning_tokens)

        greedy_times.append(greedy_seconds)
        reasoning_times.append(reasoning_seconds)

        row = {
            "id": example["id"],
            "category": example["category"],
            "state": example["state"],
            "question": example["question"],
            "target": target,
            "direct": {
                "selected": direct_choice,
                "probability_true": direct["raw_probability_true"],
                "score_delta": direct["score_delta"],
                "correct": direct_choice == target,
                "generated_output_tokens": 0,
            },
            "greedy": {
                "text": greedy_text,
                "selected": greedy_choice,
                "parsed": greedy_choice is not None,
                "correct": (
                    greedy_choice is not None
                    and greedy_choice == target
                ),
                "generated_output_tokens": greedy_tokens,
                "seconds": greedy_seconds,
            },
            "reasoning": {
                "text": reasoning_text,
                "selected": reasoning_choice,
                "parsed": reasoning_choice is not None,
                "correct": (
                    reasoning_choice is not None
                    and reasoning_choice == target
                ),
                "generated_output_tokens": reasoning_tokens,
                "seconds": reasoning_seconds,
            },
        }

        rows.append(row)

        print(
            f"{index:3d}/120  "
            f'{example["id"]:<40} '
            f"target={str(target):5}  "
            f"direct={str(direct_choice):5}  "
            f"greedy={str(greedy_choice):5}  "
            f"reason={str(reasoning_choice):5}"
        )

    direct_accuracy = strict_accuracy(
        direct_selected,
        targets,
    )

    greedy_accuracy = strict_accuracy(
        greedy_selected,
        targets,
    )

    reasoning_accuracy = strict_accuracy(
        reasoning_selected,
        targets,
    )

    greedy_parsed = sum(
        value is not None
        for value in greedy_selected
    )

    reasoning_parsed = sum(
        value is not None
        for value in reasoning_selected
    )

    dg_matches, dg_count, dg_agreement = agreement(
        direct_selected,
        greedy_selected,
    )

    dr_matches, dr_count, dr_agreement = agreement(
        direct_selected,
        reasoning_selected,
    )

    gr_matches, gr_count, gr_agreement = agreement(
        greedy_selected,
        reasoning_selected,
    )

    reasoning_fixes = sum(
        direct != target
        and reasoning is not None
        and reasoning == target
        for direct, reasoning, target in zip(
            direct_selected,
            reasoning_selected,
            targets,
        )
    )

    reasoning_breaks = sum(
        direct == target
        and reasoning is not None
        and reasoning != target
        for direct, reasoning, target in zip(
            direct_selected,
            reasoning_selected,
            targets,
        )
    )

    category_rows = []

    categories = sorted(
        {example["category"] for example in examples}
    )

    for category in categories:
        indexes = [
            index
            for index, example in enumerate(examples)
            if example["category"] == category
        ]

        category_targets = [
            targets[index]
            for index in indexes
        ]

        category_direct = [
            direct_selected[index]
            for index in indexes
        ]

        category_greedy = [
            greedy_selected[index]
            for index in indexes
        ]

        category_reasoning = [
            reasoning_selected[index]
            for index in indexes
        ]

        category_rows.append(
            {
                "category": category,
                "count": len(indexes),
                "direct_accuracy": strict_accuracy(
                    category_direct,
                    category_targets,
                ),
                "greedy_accuracy": strict_accuracy(
                    category_greedy,
                    category_targets,
                ),
                "reasoning_accuracy": strict_accuracy(
                    category_reasoning,
                    category_targets,
                ),
            }
        )

    result = {
        "model": MODEL_ID,
        "split": "calibration",
        "example_count": len(examples),
        "test_split_used": False,
        "summary": {
            "direct_accuracy": direct_accuracy,
            "greedy_accuracy": greedy_accuracy,
            "reasoning_accuracy": reasoning_accuracy,
            "greedy_parse_rate": greedy_parsed / len(examples),
            "reasoning_parse_rate": reasoning_parsed / len(examples),
            "direct_greedy_agreement": dg_agreement,
            "direct_reasoning_agreement": dr_agreement,
            "greedy_reasoning_agreement": gr_agreement,
            "reasoning_fixes_direct_errors": reasoning_fixes,
            "reasoning_breaks_direct_correct": reasoning_breaks,
            "greedy_generated_tokens_total": sum(
                greedy_token_counts
            ),
            "reasoning_generated_tokens_total": sum(
                reasoning_token_counts
            ),
            "greedy_generated_tokens_mean": statistics.mean(
                greedy_token_counts
            ),
            "reasoning_generated_tokens_mean": statistics.mean(
                reasoning_token_counts
            ),
            "greedy_latency_median_ms": (
                statistics.median(greedy_times) * 1000.0
            ),
            "reasoning_latency_median_ms": (
                statistics.median(reasoning_times) * 1000.0
            ),
        },
        "agreements": {
            "direct_vs_greedy": {
                "matches": dg_matches,
                "comparable": dg_count,
            },
            "direct_vs_reasoning": {
                "matches": dr_matches,
                "comparable": dr_count,
            },
            "greedy_vs_reasoning": {
                "matches": gr_matches,
                "comparable": gr_count,
            },
        },
        "categories": category_rows,
        "examples": rows,
    }

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print("=== OVERALL ===")
    print(
        f"Direct accuracy:    "
        f"{direct_accuracy:.6f}"
    )
    print(
        f"Greedy accuracy:    "
        f"{greedy_accuracy:.6f}"
    )
    print(
        f"Reasoning accuracy: "
        f"{reasoning_accuracy:.6f}"
    )
    print()

    print(
        f"Greedy parsed:       "
        f"{greedy_parsed}/120"
    )
    print(
        f"Reasoning parsed:    "
        f"{reasoning_parsed}/120"
    )
    print()

    print(
        "Direct/greedy agreement:   "
        f"{dg_matches}/{dg_count} "
        f"({dg_agreement:.3%})"
    )

    print(
        "Direct/reasoning agreement:"
        f" {dr_matches}/{dr_count} "
        f"({dr_agreement:.3%})"
    )

    print(
        "Greedy/reasoning agreement:"
        f" {gr_matches}/{gr_count} "
        f"({gr_agreement:.3%})"
    )

    print()
    print(
        "Reasoning fixes direct errors:",
        reasoning_fixes,
    )
    print(
        "Reasoning breaks direct correct:",
        reasoning_breaks,
    )

    print()
    print("=== BY CATEGORY ===")
    print(
        f'{"Category":<16}'
        f'{"Direct":>10}'
        f'{"Greedy":>10}'
        f'{"Reasoning":>12}'
    )

    for row in category_rows:
        print(
            f'{row["category"]:<16}'
            f'{row["direct_accuracy"]:>10.3f}'
            f'{row["greedy_accuracy"]:>10.3f}'
            f'{row["reasoning_accuracy"]:>12.3f}'
        )

    print()
    print("=== GENERATION COST ===")
    print(
        "Greedy output tokens total:",
        sum(greedy_token_counts),
    )
    print(
        "Greedy output tokens mean: ",
        f"{statistics.mean(greedy_token_counts):.2f}",
    )
    print(
        "Greedy median latency:      ",
        f"{statistics.median(greedy_times) * 1000.0:.2f} ms",
    )

    print()
    print(
        "Reasoning output tokens total:",
        sum(reasoning_token_counts),
    )
    print(
        "Reasoning output tokens mean: ",
        f"{statistics.mean(reasoning_token_counts):.2f}",
    )
    print(
        "Reasoning median latency:      ",
        f"{statistics.median(reasoning_times) * 1000.0:.2f} ms",
    )

    print()
    print("Saved:", OUTPUT)

    print()
    print("=== NEXT DECISION ===")

    if reasoning_accuracy >= direct_accuracy + 0.15:
        print(
            "Reasoning improved accuracy substantially. "
            "This suggests reasoning rollout is providing "
            "useful computation that one-step direct scoring "
            "does not perform."
        )
    elif reasoning_accuracy <= 0.65:
        print(
            "Reasoning remains near the direct baseline. "
            "Next control: run the same calibration benchmark "
            "with Qwen/Qwen2.5-3B-Instruct."
        )
    else:
        print(
            "Reasoning improved somewhat but not decisively. "
            "Inspect category-level results and disagreements "
            "before deciding whether to test a larger model."
        )


if __name__ == "__main__":
    main()
