from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TokenizedContinuation:
    input_ids: list[int]
    prefix_length: int
    target_token_ids: list[int]


def tokenize_continuation(
    tokenizer: object,
    prefix: str,
    continuation: str,
) -> TokenizedContinuation:
    if not prefix:
        raise ValueError("prefix must not be empty.")

    if not continuation:
        raise ValueError("continuation must not be empty.")

    prefix_ids = tokenizer.encode(
        prefix,
        add_special_tokens=False,
    )

    full_ids = tokenizer.encode(
        prefix + continuation,
        add_special_tokens=False,
    )

    if full_ids[: len(prefix_ids)] != prefix_ids:
        raise ValueError(
            "Tokenizer boundary changed the prefix tokenization. "
            "Use a continuation boundary that preserves the prefix tokens."
        )

    target_token_ids = full_ids[len(prefix_ids) :]

    if not target_token_ids:
        raise ValueError("continuation produced no target tokens.")

    return TokenizedContinuation(
        input_ids=full_ids,
        prefix_length=len(prefix_ids),
        target_token_ids=target_token_ids,
    )
