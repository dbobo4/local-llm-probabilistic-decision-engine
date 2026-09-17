import pytest
import torch

from llm_decision_engine.scoring import (
    normalize_candidate_scores,
    score_sequence_log_likelihood,
)


def test_sequence_log_likelihood_sum_and_mean():
    token_logits = torch.tensor([
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 2.0],
    ])

    target_token_ids = [1, 2]

    summed = score_sequence_log_likelihood(
        token_logits,
        target_token_ids,
        reduction="sum",
    )
    mean = score_sequence_log_likelihood(
        token_logits,
        target_token_ids,
        reduction="mean",
    )

    assert summed == pytest.approx(-0.47908953, rel=1e-6)
    assert mean == pytest.approx(-0.23954477, rel=1e-6)


def test_normalize_candidate_scores():
    probabilities = normalize_candidate_scores([0.0, 2.0])

    assert probabilities[0] == pytest.approx(0.11920292, rel=1e-6)
    assert probabilities[1] == pytest.approx(0.88079708, rel=1e-6)
    assert sum(probabilities) == pytest.approx(1.0, rel=1e-6)


def test_sequence_scoring_rejects_misaligned_lengths():
    token_logits = torch.zeros((2, 3))

    with pytest.raises(ValueError, match="sequence length"):
        score_sequence_log_likelihood(
            token_logits,
            [1],
        )
