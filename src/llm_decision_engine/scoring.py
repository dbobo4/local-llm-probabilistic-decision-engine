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
