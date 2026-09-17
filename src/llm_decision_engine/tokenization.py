from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class TokenizedContinuation:
    input_ids: list[int]
    prefix_length: int
    target_token_ids: list[int]


@dataclass(frozen=True, slots=True)
class BatchedContinuations:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    prefix_lengths: list[int]
    target_token_ids: list[list[int]]


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

    if not prefix_ids:
        raise ValueError("prefix produced no tokens.")

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


def batch_continuations(
    continuations: list[TokenizedContinuation],
    *,
    pad_token_id: int,
    device: torch.device | str | None = None,
) -> BatchedContinuations:
    if not continuations:
        raise ValueError("continuations must not be empty.")

    if pad_token_id < 0:
        raise ValueError("pad_token_id must be non-negative.")

    max_length = max(len(item.input_ids) for item in continuations)

    input_ids = torch.full(
        (len(continuations), max_length),
        pad_token_id,
        dtype=torch.long,
        device=device,
    )

    attention_mask = torch.zeros(
        (len(continuations), max_length),
        dtype=torch.long,
        device=device,
    )

    for row, item in enumerate(continuations):
        length = len(item.input_ids)
        input_ids[row, :length] = torch.tensor(
            item.input_ids,
            dtype=torch.long,
            device=device,
        )
        attention_mask[row, :length] = 1

    return BatchedContinuations(
        input_ids=input_ids,
        attention_mask=attention_mask,
        prefix_lengths=[item.prefix_length for item in continuations],
        target_token_ids=[item.target_token_ids for item in continuations],
    )
