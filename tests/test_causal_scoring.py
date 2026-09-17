import pytest
import torch

from llm_decision_engine.scoring import score_causal_continuation


def test_causal_continuation_uses_previous_token_positions():
    logits = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.0, 4.0, 0.0],
        [0.0, 0.0, 4.0],
        [4.0, 0.0, 0.0],
    ])

    score = score_causal_continuation(
        logits,
        prefix_length=2,
        target_token_ids=[1, 2],
    )

    expected = 2 * torch.log_softmax(
        torch.tensor([0.0, 4.0, 0.0]),
        dim=-1,
    )[1].item()

    assert score == pytest.approx(expected, abs=1e-6)


def test_causal_continuation_rejects_too_short_logits():
    logits = torch.zeros((2, 3))

    with pytest.raises(ValueError, match="too short"):
        score_causal_continuation(
            logits,
            prefix_length=2,
            target_token_ids=[1, 2],
        )

