from __future__ import annotations

import json
from pathlib import Path

import torch

from llm_decision_engine import DecisionEngine
from llm_decision_engine.tokenization import tokenize_continuation


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "benchmarks" / "data" / "binary_quality_v1.json"


def make_prefix(engine, state: str, question: str) -> str:
    user_prompt = f"""STATE:
{state}

QUESTION:
{question}

Answer exactly one word: True or False."""

    return engine.tokenizer.apply_chat_template(
        [{"role": "user", "content": user_prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def main() -> None:
    examples = json.loads(DATASET.read_text(encoding="utf-8"))
    engine = DecisionEngine(model="Qwen/Qwen2.5-1.5B-Instruct")
    device = next(engine.model.parameters()).device

    masses = []
    normalized_true_values = []

    print(
        f"{'id':>12}  "
        f"{'Praw(True)':>11}  "
        f"{'Praw(False)':>12}  "
        f"{'mass':>10}  "
        f"{'Prestricted(True)':>17}"
    )

    for example in examples:
        prefix = make_prefix(
            engine,
            example["state"],
            example["question"],
        )

        true_tokens = tokenize_continuation(
            engine.tokenizer,
            prefix,
            "True",
        )
        false_tokens = tokenize_continuation(
            engine.tokenizer,
            prefix,
            "False",
        )

        if len(true_tokens.target_token_ids) != 1:
            raise RuntimeError("True must be one token.")

        if len(false_tokens.target_token_ids) != 1:
            raise RuntimeError("False must be one token.")

        if true_tokens.input_ids[:true_tokens.prefix_length] != (
            false_tokens.input_ids[:false_tokens.prefix_length]
        ):
            raise RuntimeError("Candidate prefixes differ.")

        prefix_ids = true_tokens.input_ids[:true_tokens.prefix_length]

        input_ids = torch.tensor(
            [prefix_ids],
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.ones_like(input_ids)

        with torch.inference_mode():
            outputs = engine.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        next_token_logits = outputs.logits[0, -1].float()
        vocabulary_probabilities = torch.softmax(
            next_token_logits,
            dim=-1,
        )

        true_id = true_tokens.target_token_ids[0]
        false_id = false_tokens.target_token_ids[0]

        raw_true = vocabulary_probabilities[true_id].item()
        raw_false = vocabulary_probabilities[false_id].item()

        candidate_mass = raw_true + raw_false
        restricted_true = raw_true / candidate_mass

        masses.append(candidate_mass)
        normalized_true_values.append(restricted_true)

        print(
            f'{example["id"]:>12}  '
            f"{raw_true:11.6f}  "
            f"{raw_false:12.6f}  "
            f"{candidate_mass:10.6f}  "
            f"{restricted_true:17.6f}"
        )

    print()
    print("--- Candidate mass summary ---")
    print(f"Examples:    {len(masses)}")
    print(f"Mean mass:   {sum(masses) / len(masses):.6f}")
    print(f"Min mass:    {min(masses):.6f}")
    print(f"Max mass:    {max(masses):.6f}")


if __name__ == "__main__":
    main()
