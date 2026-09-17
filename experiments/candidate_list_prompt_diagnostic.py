from __future__ import annotations

import json
from pathlib import Path

import torch

from llm_decision_engine import DecisionEngine
from llm_decision_engine.scoring import (
    normalize_candidate_scores,
    score_causal_continuation,
)
from llm_decision_engine.tokenization import tokenize_continuation


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "benchmarks" / "data" / "binary_quality_v1.json"


def score_without_candidate_list(engine, state: str, question: str):
    user_prompt = f"""STATE:
{state}

QUESTION:
{question}

Answer the question directly."""

    prefix = engine.tokenizer.apply_chat_template(
        [{"role": "user", "content": user_prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )

    candidates = ["yes", "no"]
    scores = []
    device = next(engine.model.parameters()).device

    for candidate in candidates:
        tokenized = tokenize_continuation(
            engine.tokenizer,
            prefix,
            candidate,
        )

        input_ids = torch.tensor(
            [tokenized.input_ids],
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.ones_like(input_ids)

        with torch.inference_mode():
            outputs = engine.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        scores.append(
            score_causal_continuation(
                outputs.logits[0],
                tokenized.prefix_length,
                tokenized.target_token_ids,
                reduction="sum",
            )
        )

    probabilities = normalize_candidate_scores(scores)

    return {
        "yes": probabilities[0],
        "no": probabilities[1],
    }


def main() -> None:
    examples = json.loads(DATASET.read_text(encoding="utf-8"))
    engine = DecisionEngine(model="Qwen/Qwen2.5-1.5B-Instruct")

    current_correct = 0
    no_list_correct = 0
    changed = 0
    deltas = []

    print(
        f"{'id':>12}  {'target':>6}  "
        f"{'Pyes current':>12}  {'Pyes no-list':>13}  "
        f"{'delta':>10}  {'cur':>5}  {'new':>5}"
    )

    for example in examples:
        current = engine.boolean(
            state=example["state"],
            question=example["question"],
            execution="sequential",
        )

        no_list = score_without_candidate_list(
            engine,
            example["state"],
            example["question"],
        )

        p_current = current.probability_true
        p_no_list = no_list["yes"]

        selected_current = p_current >= 0.5
        selected_no_list = p_no_list >= 0.5
        target = example["target"]

        current_correct += selected_current == target
        no_list_correct += selected_no_list == target
        changed += selected_current != selected_no_list
        deltas.append(abs(p_no_list - p_current))

        print(
            f'{example["id"]:>12}  '
            f"{str(target):>6}  "
            f"{p_current:12.6f}  "
            f"{p_no_list:13.6f}  "
            f"{p_no_list - p_current:+10.6f}  "
            f"{str(selected_current):>5}  "
            f"{str(selected_no_list):>5}"
        )

    n = len(examples)

    print()
    print("--- Prompt strategy summary ---")
    print(f"Accuracy current:      {current_correct / n:.6f}")
    print(f"Accuracy no-list:      {no_list_correct / n:.6f}")
    print(f"Selections changed:    {changed}/{n}")
    print(f"Mean absolute delta:   {sum(deltas) / n:.6f}")
    print(f"Max absolute delta:    {max(deltas):.6f}")


if __name__ == "__main__":
    main()
