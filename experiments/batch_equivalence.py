import torch

from llm_decision_engine.models import load_model
from llm_decision_engine.scoring import (
    normalize_candidate_scores,
    score_causal_continuation,
)
from llm_decision_engine.tokenization import (
    batch_continuations,
    tokenize_continuation,
)


MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
candidates = ["technical support", "billing", "sales"]

bundle = load_model(MODEL_ID)
tokenizer = bundle.tokenizer
model = bundle.model
device = next(model.parameters()).device

state = "The customer account has not worked for three days and they are losing sales."
question = "Which department should handle this?"
candidate_list = "\n".join(f"- {candidate}" for candidate in candidates)

user_prompt = f"""STATE:
{state}

QUESTION:
{question}

CANDIDATES:
{candidate_list}

Return exactly one candidate."""

prefix = tokenizer.apply_chat_template(
    [{"role": "user", "content": user_prompt}],
    tokenize=False,
    add_generation_prompt=True,
)

items = [
    tokenize_continuation(tokenizer, prefix, candidate)
    for candidate in candidates
]

print("Model dtype:", next(model.parameters()).dtype)
print("Attention implementation:", getattr(model.config, "_attn_implementation", "unknown"))
print()

sequential_scores = []

for item in items:
    input_ids = torch.tensor([item.input_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)

    with torch.inference_mode():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    sequential_scores.append(
        score_causal_continuation(
            outputs.logits[0],
            item.prefix_length,
            item.target_token_ids,
            reduction="sum",
        )
    )

pad_token_id = tokenizer.pad_token_id
if pad_token_id is None:
    pad_token_id = tokenizer.eos_token_id

batch = batch_continuations(
    items,
    pad_token_id=pad_token_id,
    device=device,
)

with torch.inference_mode():
    outputs = model(
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
    )

batched_scores = [
    score_causal_continuation(
        outputs.logits[row],
        batch.prefix_lengths[row],
        batch.target_token_ids[row],
        reduction="sum",
    )
    for row in range(len(candidates))
]

sequential_probabilities = normalize_candidate_scores(sequential_scores)
batched_probabilities = normalize_candidate_scores(batched_scores)

print("Score comparison:")
for candidate, sequential, batched in zip(
    candidates, sequential_scores, batched_scores
):
    print(
        f"{candidate:<20} "
        f"sequential={sequential: .9f} "
        f"batched={batched: .9f} "
        f"delta={batched - sequential:+.9f}"
    )

print("\nProbability comparison:")
for candidate, sequential, batched in zip(
    candidates, sequential_probabilities, batched_probabilities
):
    print(
        f"{candidate:<20} "
        f"sequential={sequential:.9f} "
        f"batched={batched:.9f} "
        f"delta={batched - sequential:+.9f}"
    )

print("\nBatch input IDs shape:", tuple(batch.input_ids.shape))
print("Batch attention masks:")
print(batch.attention_mask.cpu())
