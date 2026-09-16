import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

CANDIDATES = {
    "A": "technical support",
    "B": "billing",
    "C": "sales",
}


def main():
    print(f"Loading model: {MODEL_ID}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    model.eval()

    user_prompt = """STATE:
The customer's account has not worked for three days and they are losing sales.

QUESTION:
Which department should handle this?

OPTIONS:
A = technical support
B = billing
C = sales

Answer with exactly one label: A, B, or C."""

    messages = [
        {
            "role": "user",
            "content": user_prompt,
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(model.device)

    print("\nCandidate tokenization:")

    candidate_token_ids = {}

    for label in CANDIDATES:
        token_ids = tokenizer.encode(
            label,
            add_special_tokens=False,
        )

        print(
            f"{label}: token_ids={token_ids}, "
            f"tokens={tokenizer.convert_ids_to_tokens(token_ids)}"
        )

        if len(token_ids) != 1:
            raise RuntimeError(
                f"Candidate label {label!r} is not a single token."
            )

        candidate_token_ids[label] = token_ids[0]

    with torch.inference_mode():
        outputs = model(**inputs)

    # Shape:
    # [batch_size, sequence_length, vocabulary_size]
    all_logits = outputs.logits

    # We only need the prediction after the final input token.
    next_token_logits = all_logits[0, -1, :]

    labels = list(CANDIDATES.keys())

    candidate_logits = torch.stack(
        [
            next_token_logits[candidate_token_ids[label]]
            for label in labels
        ]
    )

    # Probability over the complete model vocabulary.
    full_vocab_probs = torch.softmax(
        next_token_logits.float(),
        dim=-1,
    )

    # Probability after restricting the output space to A/B/C.
    candidate_probs = torch.softmax(
        candidate_logits.float(),
        dim=-1,
    )

    print("\nRaw candidate logits:")

    for label, logit in zip(labels, candidate_logits):
        print(
            f"{label} ({CANDIDATES[label]}): "
            f"{logit.item():.6f}"
        )

    print("\nFull-vocabulary probabilities:")

    for label in labels:
        token_id = candidate_token_ids[label]

        print(
            f"{label} ({CANDIDATES[label]}): "
            f"{full_vocab_probs[token_id].item():.8f}"
        )

    print("\nRestricted candidate probabilities:")

    for label, probability in zip(labels, candidate_probs):
        print(
            f"{label} ({CANDIDATES[label]}): "
            f"{probability.item():.6f}"
        )

    best_index = torch.argmax(candidate_probs).item()
    best_label = labels[best_index]

    print("\nSelected decision:")
    print(
        f"{best_label} -> {CANDIDATES[best_label]} "
        f"({candidate_probs[best_index].item():.2%})"
    )

    print("\nGenerated output tokens: 0")


if __name__ == "__main__":
    main()