import pytest
import torch

from llm_decision_engine.scoring import score_single_token_candidates


def test_single_token_candidate_scoring():
    next_token_logits = torch.tensor([0.0, 1.0, 2.0, 3.0])
    candidate_token_ids = [1, 3]

    logits, probabilities, candidate_mass = score_single_token_candidates(
        next_token_logits,
        candidate_token_ids,
    )

    assert logits.tolist() == [1.0, 3.0]
    assert probabilities[0].item() == pytest.approx(0.11920292, rel=1e-6)
    assert probabilities[1].item() == pytest.approx(0.88079708, rel=1e-6)

    full_probs = torch.softmax(next_token_logits, dim=-1)
    expected_mass = full_probs[1].item() + full_probs[3].item()
    assert candidate_mass == pytest.approx(expected_mass, rel=1e-6)

    assert probabilities.sum().item() == pytest.approx(1.0, rel=1e-6)
