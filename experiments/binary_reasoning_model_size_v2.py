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

OUTPUT = (
    ROOT
    / "results"
    / "binary_reasoning_model_size_v2_calibration.json"
)

EXPECTED_DATASET_SHA256 = (
    "d2116a228b48c959f3c77e8e4ecfa816"
    "033aa4044ec55bbfb930553985a2e4b8"
)

MODEL_IDS = (
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
)

REASONING_MAX_NEW_TOKENS = 128


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def make_binary_prefix(
    tokenizer,
    state: str,
    question: str,
) -> str:
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


def make_reasoning_prefix(
    tokenizer,
    state: str,
    question: str,
) -> str:
    content = (
        f"STATE:\n{state}\n\n"
        f"QUESTION:\n{question}\n\n"
        "Work out whether the statement is correct. "
        "Use only the necessary arithmetic or logical steps. "
        "Keep the analysis brief, at most three short sentences. "
        "Do not give a final True or False label yet."
    )

    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )


def generate_text(
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


def parse_one_word_binary(text: str) -> bool | None:
    match = re.match(
        r"^\s*(True|False)\b",
        text,
        flags=re.IGNORECASE,
    )

    if match is None:
        return None

    return match.group(1).lower() == "true"


def accuracy(
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


def run_model(
    model_id: str,
    examples: list[dict],
) -> dict:
    print()
    print("=" * 80)
    print("MODEL:", model_id)
    print("=" * 80)

    engine = DecisionEngine(model=model_id)

    targets = []

    direct_choices = []
    greedy_choices = []

    reasoned_direct_choices = []
    reasoned_greedy_choices = []

    rationale_token_counts = []
    rationale_times = []

    rows = []

    for index, example in enumerate(examples, start=1):
        target = example["target"]

        direct = engine.boolean(
            state=example["state"],
            question=example["question"],
            scoring="sum",
            execution="sequential",
        )

        binary_prefix = make_binary_prefix(
            engine.tokenizer,
            example["state"],
            example["question"],
        )

        greedy_text, greedy_tokens, greedy_seconds = generate_text(
            engine,
            binary_prefix,
            max_new_tokens=4,
        )

        greedy_choice = parse_one_word_binary(
            greedy_text
        )

        reasoning_prefix = make_reasoning_prefix(
            engine.tokenizer,
            example["state"],
            example["question"],
        )

        rationale, rationale_tokens, rationale_seconds = generate_text(
            engine,
            reasoning_prefix,
            max_new_tokens=REASONING_MAX_NEW_TOKENS,
        )

        reasoning_state = (
            f"{example['state']}\n\n"
            "MODEL ANALYSIS:\n"
            f"{rationale}"
        )

        reasoned_direct = engine.boolean(
            state=reasoning_state,
            question=example["question"],
            scoring="sum",
            execution="sequential",
        )

        reasoned_binary_prefix = make_binary_prefix(
            engine.tokenizer,
            reasoning_state,
            example["question"],
        )

        (
            reasoned_greedy_text,
            reasoned_greedy_tokens,
            reasoned_greedy_seconds,
        ) = generate_text(
            engine,
            reasoned_binary_prefix,
            max_new_tokens=4,
        )

        reasoned_greedy_choice = parse_one_word_binary(
            reasoned_greedy_text
        )

        targets.append(target)

        direct_choices.append(
            direct.selected
        )

        greedy_choices.append(
            greedy_choice
        )

        reasoned_direct_choices.append(
            reasoned_direct.selected
        )

        reasoned_greedy_choices.append(
            reasoned_greedy_choice
        )

        rationale_token_counts.append(
            rationale_tokens
        )

        rationale_times.append(
            rationale_seconds
        )

        rows.append(
            {
                "id": example["id"],
                "category": example["category"],
                "state": example["state"],
                "question": example["question"],
                "target": target,
                "direct": {
                    "selected": direct.selected,
                    "probability_true": direct.probability_true,
                    "score_delta": (
                        direct.scores["yes"]
                        - direct.scores["no"]
                    ),
                    "correct": direct.selected == target,
                    "generated_output_tokens": 0,
                },
                "greedy": {
                    "text": greedy_text,
                    "selected": greedy_choice,
                    "correct": (
                        greedy_choice is not None
                        and greedy_choice == target
                    ),
                    "generated_output_tokens": greedy_tokens,
                    "seconds": greedy_seconds,
                },
                "reasoning": {
                    "text": rationale,
                    "generated_output_tokens": rationale_tokens,
                    "seconds": rationale_seconds,
                    "hit_token_limit": (
                        rationale_tokens
                        >= REASONING_MAX_NEW_TOKENS
                    ),
                },
                "reasoning_conditioned_direct": {
                    "selected": reasoned_direct.selected,
                    "probability_true": (
                        reasoned_direct.probability_true
                    ),
                    "score_delta": (
                        reasoned_direct.scores["yes"]
                        - reasoned_direct.scores["no"]
                    ),
                    "correct": (
                        reasoned_direct.selected == target
                    ),
                    "generated_output_tokens": 0,
                },
                "reasoning_conditioned_greedy": {
                    "text": reasoned_greedy_text,
                    "selected": reasoned_greedy_choice,
                    "correct": (
                        reasoned_greedy_choice is not None
                        and reasoned_greedy_choice == target
                    ),
                    "generated_output_tokens": (
                        reasoned_greedy_tokens
                    ),
                    "seconds": reasoned_greedy_seconds,
                },
            }
        )

        print(
            f"{index:3d}/{len(examples)}  "
            f'{example["id"]:<40}  '
            f"target={str(target):5}  "
            f"direct={str(direct.selected):5}  "
            f"greedy={str(greedy_choice):5}  "
            f"reasoned={str(reasoned_direct.selected):5}  "
            f"reasoned_greedy={str(reasoned_greedy_choice):5}"
        )

    direct_accuracy = accuracy(
        direct_choices,
        targets,
    )

    greedy_accuracy = accuracy(
        greedy_choices,
        targets,
    )

    reasoned_direct_accuracy = accuracy(
        reasoned_direct_choices,
        targets,
    )

    reasoned_greedy_accuracy = accuracy(
        reasoned_greedy_choices,
        targets,
    )

    dg_matches, dg_count, dg_rate = agreement(
        direct_choices,
        greedy_choices,
    )

    rg_matches, rg_count, rg_rate = agreement(
        reasoned_direct_choices,
        reasoned_greedy_choices,
    )

    reasoning_fixes = sum(
        direct != target
        and reasoned == target
        for direct, reasoned, target in zip(
            direct_choices,
            reasoned_direct_choices,
            targets,
        )
    )

    reasoning_breaks = sum(
        direct == target
        and reasoned != target
        for direct, reasoned, target in zip(
            direct_choices,
            reasoned_direct_choices,
            targets,
        )
    )

    categories = sorted(
        {
            example["category"]
            for example in examples
        }
    )

    category_results = []

    for category in categories:
        indexes = [
            i
            for i, example in enumerate(examples)
            if example["category"] == category
        ]

        category_targets = [
            targets[i]
            for i in indexes
        ]

        category_direct = [
            direct_choices[i]
            for i in indexes
        ]

        category_reasoned = [
            reasoned_direct_choices[i]
            for i in indexes
        ]

        category_results.append(
            {
                "category": category,
                "count": len(indexes),
                "direct_accuracy": accuracy(
                    category_direct,
                    category_targets,
                ),
                "reasoning_conditioned_accuracy": accuracy(
                    category_reasoned,
                    category_targets,
                ),
            }
        )

    result = {
        "model": model_id,
        "direct_accuracy": direct_accuracy,
        "greedy_accuracy": greedy_accuracy,
        "reasoning_conditioned_direct_accuracy": (
            reasoned_direct_accuracy
        ),
        "reasoning_conditioned_greedy_accuracy": (
            reasoned_greedy_accuracy
        ),
        "direct_greedy_agreement": {
            "matches": dg_matches,
            "comparable": dg_count,
            "rate": dg_rate,
        },
        "reasoned_direct_greedy_agreement": {
            "matches": rg_matches,
            "comparable": rg_count,
            "rate": rg_rate,
        },
        "reasoning_fixes_direct_errors": reasoning_fixes,
        "reasoning_breaks_direct_correct": reasoning_breaks,
        "rationale_generated_tokens_total": sum(
            rationale_token_counts
        ),
        "rationale_generated_tokens_mean": statistics.mean(
            rationale_token_counts
        ),
        "rationale_median_latency_ms": (
            statistics.median(rationale_times) * 1000.0
        ),
        "rationale_hit_token_limit": sum(
            tokens >= REASONING_MAX_NEW_TOKENS
            for tokens in rationale_token_counts
        ),
        "categories": category_results,
        "examples": rows,
    }

    print()
    print("--- SUMMARY ---")
    print(
        f"Direct accuracy:                 "
        f"{direct_accuracy:.6f}"
    )
    print(
        f"Greedy accuracy:                 "
        f"{greedy_accuracy:.6f}"
    )
    print(
        f"Reasoning-conditioned direct:    "
        f"{reasoned_direct_accuracy:.6f}"
    )
    print(
        f"Reasoning-conditioned greedy:    "
        f"{reasoned_greedy_accuracy:.6f}"
    )

    print()
    print(
        "Direct / greedy agreement:      "
        f"{dg_matches}/{dg_count} "
        f"({dg_rate:.3%})"
    )

    print(
        "Reasoned direct / greedy:       "
        f"{rg_matches}/{rg_count} "
        f"({rg_rate:.3%})"
    )

    print()
    print(
        "Reasoning fixes direct errors:  ",
        reasoning_fixes,
    )
    print(
        "Reasoning breaks direct correct:",
        reasoning_breaks,
    )

    print()
    print(
        "Reasoning token mean:           "
        f"{statistics.mean(rationale_token_counts):.2f}"
    )
    print(
        "Reasoning median latency:       "
        f"{statistics.median(rationale_times) * 1000.0:.2f} ms"
    )
    print(
        "Reasoning hit token limit:      "
        f"{sum(tokens >= REASONING_MAX_NEW_TOKENS for tokens in rationale_token_counts)}"
    )

    print()
    print("--- BY CATEGORY ---")
    print(
        f'{"Category":<16}'
        f'{"Direct":>10}'
        f'{"Reasoned":>12}'
        f'{"Delta":>10}'
    )

    for row in category_results:
        delta = (
            row["reasoning_conditioned_accuracy"]
            - row["direct_accuracy"]
        )

        print(
            f'{row["category"]:<16}'
            f'{row["direct_accuracy"]:>10.3f}'
            f'{row["reasoning_conditioned_accuracy"]:>12.3f}'
            f'{delta:>+10.3f}'
        )

    del engine
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def main() -> None:
    dataset_hash = sha256_file(DATASET)

    if dataset_hash != EXPECTED_DATASET_SHA256:
        raise RuntimeError(
            "Calibration dataset hash mismatch. "
            f"Expected {EXPECTED_DATASET_SHA256}, "
            f"got {dataset_hash}."
        )

    examples = json.loads(
        DATASET.read_text(encoding="utf-8")
    )

    if len(examples) != 120:
        raise RuntimeError(
            f"Expected 120 examples, got {len(examples)}."
        )

    results = []

    for model_id in MODEL_IDS:
        results.append(
            run_model(
                model_id,
                examples,
            )
        )

    payload = {
        "split": "calibration",
        "dataset": str(
            DATASET.relative_to(ROOT)
        ),
        "dataset_sha256": dataset_hash,
        "test_split_used": False,
        "reasoning_max_new_tokens": (
            REASONING_MAX_NEW_TOKENS
        ),
        "models": results,
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
    print("FINAL MODEL COMPARISON")
    print("=" * 80)

    print(
        f'{"Model":<34}'
        f'{"Direct":>10}'
        f'{"Reasoned":>12}'
        f'{"Gain":>10}'
    )

    for result in results:
        gain = (
            result[
                "reasoning_conditioned_direct_accuracy"
            ]
            - result["direct_accuracy"]
        )

        print(
            f'{result["model"]:<34}'
            f'{result["direct_accuracy"]:>10.3f}'
            f'{result["reasoning_conditioned_direct_accuracy"]:>12.3f}'
            f'{gain:>+10.3f}'
        )

    print()
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()
