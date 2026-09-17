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


def make_prompt(state: str, question: str) -> str:
    return f"""STATE:
{state}

QUESTION:
{question}

Answer exactly one word: True or False."""


def make_prefix(engine, state: str, question: str) -> str:
    return engine.tokenizer.apply_chat_template(
        [{"role": "user", "content": make_prompt(state, question)}],
        tokenize=False,
        add_generation_prompt=True,
    )


def direct_probability_true(engine, state: str, question: str) -> float:
    prefix = make_prefix(engine, state, question)
    device = next(engine.model.parameters()).device
    scores = []

    for candidate in ["True", "False"]:
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


def greedy_generate(engine, state: str, question: str) -> str:
    prefix = make_prefix(engine, state, question)
    device = next(engine.model.parameters()).device

    encoded = engine.tokenizer(
        prefix,
        return_tensors="pt",
        add_special_tokens=False,
    )

    encoded = {
        key: value.to(device)
        for key, value in encoded.items()
    }

    input_length = encoded["input_ids"].shape[1]

    with torch.inference_mode():
        output_ids = engine.model.generate(
            **encoded,
            max_new_tokens=4,
            do_sample=False,
            pad_token_id=engine.tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0, input_length:]

    return engine.tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()

def parse_boolean(text: str):
    normalized = text.strip().lower()

    if normalized.startswith("true"):
        return True

    if normalized.startswith("false"):
        return False

    return None


def main() -> None:
    examples = json.loads(DATASET.read_text(encoding="utf-8"))
    engine = DecisionEngine(model="Qwen/Qwen2.5-1.5B-Instruct")

    direct_correct = 0
    generate_correct = 0
    agreement = 0
    unparsed = 0

    print(
        f"{'id':>12}  {'target':>6}  {'P(True)':>10}  "
        f"{'direct':>7}  {'generate':>12}  {'agree':>7}"
    )

    for example in examples:
        probability_true = direct_probability_true(
            engine,
            example["state"],
            example["question"],
        )

        direct_selected = probability_true >= 0.5

        generated_text = greedy_generate(
            engine,
            example["state"],
            example["question"],
        )

        generated_selected = parse_boolean(generated_text)
        target = example["target"]

        direct_correct += direct_selected == target

        if generated_selected is None:
            unparsed += 1
        else:
            generate_correct += generated_selected == target
            agreement += generated_selected == direct_selected

        print(
            f'{example["id"]:>12}  '
            f"{str(target):>6}  "
            f"{probability_true:10.6f}  "
            f"{str(direct_selected):>7}  "
            f"{generated_text!r:>12}  "
            f"{str(generated_selected == direct_selected):>7}"
        )

    parsed = len(examples) - unparsed

    print()
    print("--- Direct vs generation summary ---")
    print(f"Examples:             {len(examples)}")
    print(f"Direct accuracy:      {direct_correct / len(examples):.6f}")
    print(f"Generated parsed:     {parsed}/{len(examples)}")
    print(f"Generated unparsed:   {unparsed}")

    if parsed:
        print(f"Generate accuracy:    {generate_correct / parsed:.6f}")
        print(f"Direct/gen agreement: {agreement / parsed:.6f}")


if __name__ == "__main__":
    main()
