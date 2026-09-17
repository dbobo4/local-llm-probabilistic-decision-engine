import torch


def score_single_token_candidates(
    next_token_logits: torch.Tensor,
    candidate_token_ids: list[int],
) -> tuple[torch.Tensor, torch.Tensor, float]:
    candidate_logits = torch.stack(
        [next_token_logits[token_id] for token_id in candidate_token_ids]
    )

    candidate_probabilities = torch.softmax(
        candidate_logits.float(),
        dim=-1,
    )

    full_vocabulary_probabilities = torch.softmax(
        next_token_logits.float(),
        dim=-1,
    )

    candidate_mass = sum(
        full_vocabulary_probabilities[token_id].item()
        for token_id in candidate_token_ids
    )

    return candidate_logits, candidate_probabilities, candidate_mass


def score_sequence_log_likelihood(
    token_logits: torch.Tensor,
    target_token_ids: list[int],
    *,
    reduction: str = "sum",
) -> float:
    if token_logits.ndim != 2:
        raise ValueError("token_logits must have shape [sequence_length, vocabulary_size].")

    if not target_token_ids:
        raise ValueError("target_token_ids must contain at least one token.")

    if token_logits.shape[0] != len(target_token_ids):
        raise ValueError(
            "token_logits sequence length must match target_token_ids length."
        )

    if reduction not in {"sum", "mean"}:
        raise ValueError("reduction must be either 'sum' or 'mean'.")

    vocabulary_size = token_logits.shape[1]
    if any(token_id < 0 or token_id >= vocabulary_size for token_id in target_token_ids):
        raise ValueError("target_token_ids contains a token outside the vocabulary.")

    log_probabilities = torch.log_softmax(token_logits.float(), dim=-1)
    target_ids = torch.tensor(
        target_token_ids,
        dtype=torch.long,
        device=token_logits.device,
    )

    token_log_probabilities = log_probabilities.gather(
        1,
        target_ids.unsqueeze(1),
    ).squeeze(1)

    if reduction == "sum":
        return token_log_probabilities.sum().item()

    return token_log_probabilities.mean().item()


def normalize_candidate_scores(scores: list[float]) -> list[float]:
    if len(scores) < 2:
        raise ValueError("At least two candidate scores are required.")

    probabilities = torch.softmax(
        torch.tensor(scores, dtype=torch.float32),
        dim=-1,
    )

    return probabilities.tolist()
