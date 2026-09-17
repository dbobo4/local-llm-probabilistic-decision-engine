import torch

from llm_decision_engine.models import load_model
from llm_decision_engine.scoring import (
    normalize_candidate_scores,
    score_causal_continuation,
)
from llm_decision_engine.tokenization import tokenize_continuation


MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

state = (
    "The customer account has not worked for three days "
    "and they are losing sales."
)

question = "Which department should handle this?"

candidates = [
    "technical support",
    "billing",
    "sales",
]


bundle = load_model(MODEL_ID)
tokenizer = bundle.tokenizer
model = bundle.model

candidate_list = "\n".join(f"- {candidate}" for candidate in candidates)

user_prompt = f"""STATE:
{state}

QUESTION:
{question}

CANDIDATES:
{candidate_list}

Return exactly one candidate."""

messages = [{"role": "user", "content": user_prompt}]

prefix = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

input_device = next(model.parameters()).device

sum_scores = []
mean_scores = []

print("Candidate tokenization and scores:")

for candidate in candidates:
    tokenized = tokenize_continuation(
        tokenizer,
        prefix,
        candidate,
    )

    input_ids = torch.tensor(
        [tokenized.input_ids],
        dtype=torch.long,
        device=input_device,
    )
    attention_mask = torch.ones_like(input_ids)

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    logits = outputs.logits[0]

    sum_score = score_causal_continuation(
        logits,
        tokenized.prefix_length,
        tokenized.target_token_ids,
        reduction="sum",
    )

    mean_score = score_causal_continuation(
        logits,
        tokenized.prefix_length,
        tokenized.target_token_ids,
        reduction="mean",
    )

    sum_scores.append(sum_score)
    mean_scores.append(mean_score)

    tokens = tokenizer.convert_ids_to_tokens(
        tokenized.target_token_ids
    )

    print(
        f"  {candidate:<20} "
        f"tokens={tokens!r} "
        f"sum={sum_score:.6f} "
        f"mean={mean_score:.6f}"
    )

sum_probabilities = normalize_candidate_scores(sum_scores)
mean_probabilities = normalize_candidate_scores(mean_scores)

print("\nSUM-normalized probabilities:")
for candidate, probability in zip(candidates, sum_probabilities):
    print(f"  {candidate:<20} {probability:.6f}")

print("\nMEAN-normalized probabilities:")
for candidate, probability in zip(candidates, mean_probabilities):
    print(f"  {candidate:<20} {probability:.6f}")

print("\nGenerated output tokens: 0")
