from __future__ import annotations

import json
from pathlib import Path

import torch

from llm_decision_engine import DecisionEngine, evaluate_binary_predictions
from llm_decision_engine.scoring import (
    normalize_candidate_scores,
    score_causal_continuation,
)
from llm_decision_engine.tokenization import tokenize_continuation


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "benchmarks" / "data" / "binary_quality_v1.json"

LABEL_PAIRS = [
    ("yes", "no"),
    ("Yes", "No"),
    ("true", "false"),
    ("True", "False"),
    ("correct", "incorrect"),
]


def make_prefix(engine, state: str, question: str) -> str:
    user_prompt = f"""STATE:
{state}

QUESTION:
{question}

Answer the question directly."""

    return engine.tokenizer.apply_chat_template(
        [{"role": "user", "content": user_prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def score_pair(
    engine,
    state: str,
    question: str,
    true_label: str,
    false_label: str,
) -> float:
    prefix = make_prefix(engine, state, question)
    device = next(engine.model.parameters()).device
    scores = []

    for candidate in [true_label, false_label]:
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

    return normalize_candidate_scores(scores)[0]


def main() -> None:
    examples = json.loads(DATASET.read_text(encoding="utf-8"))
    engine = DecisionEngine(model="Qwen/Qwen2.5-1.5B-Instruct")

    first = examples[0]
    prefix = make_prefix(
        engine,
        first["state"],
        first["question"],
    )

    print("--- Label tokenization ---")

    for true_label, false_label in LABEL_PAIRS:
        for label in [true_label, false_label]:
            tokenized = tokenize_continuation(
                engine.tokenizer,
                prefix,
                label,
            )
            print(
                f"{label!r:12} "
                f"tokens={len(tokenized.target_token_ids)} "
                f"ids={tokenized.target_token_ids}"
            )

    print()
    print("--- Label surface benchmark ---")
    print(
        f"{'labels':>22}  "
        f"{'accuracy':>9}  "
        f"{'brier':>10}  "
        f"{'nll':>10}"
    )

    for true_label, false_label in LABEL_PAIRS:
        probabilities_true = []
        targets = []

        for example in examples:
            probability_true = score_pair(
                engine,
                example["state"],
                example["question"],
                true_label,
                false_label,
            )

            probabilities_true.append(probability_true)
            targets.append(example["target"])

        result = evaluate_binary_predictions(
            probabilities_true,
            targets,
        )

        label_name = f"{true_label}/{false_label}"

        print(
            f"{label_name:>22}  "
            f"{result.accuracy:9.6f}  "
            f"{result.mean_brier:10.6f}  "
            f"{result.mean_nll:10.6f}"
        )


if __name__ == "__main__":
    main()
