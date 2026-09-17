from types import SimpleNamespace

import pytest
import torch

from llm_decision_engine import DecisionEngine


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "P"

    def encode(self, text, add_special_tokens=False):
        mapping = {
            "P": [10, 11],
            "Palpha": [10, 11, 1, 2],
            "Pbeta": [10, 11, 3],
        }
        return mapping[text]


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.calls = 0

    def forward(self, input_ids, attention_mask):
        self.calls += 1
        batch_size, sequence_length = input_ids.shape
        logits = torch.zeros(
            (batch_size, sequence_length, 4),
            dtype=torch.float32,
            device=input_ids.device,
        )

        for row in range(batch_size):
            if input_ids[row, 2].item() == 1:
                logits[row, 1, 1] = 4.0
                logits[row, 2, 2] = 4.0
            elif input_ids[row, 2].item() == 3:
                logits[row, 1, 3] = 1.0

        return SimpleNamespace(logits=logits)


def make_engine():
    engine = DecisionEngine.__new__(DecisionEngine)
    engine.model_id = "fake"
    engine.tokenizer = FakeTokenizer()
    engine.model = FakeModel()
    return engine


def test_choice_requires_at_least_two_candidates():
    engine = DecisionEngine.__new__(DecisionEngine)

    with pytest.raises(ValueError, match="at least two candidates"):
        engine.choice(
            state="test", question="test", candidates=["one"]
        )


def test_choice_rejects_duplicate_candidates():
    engine = DecisionEngine.__new__(DecisionEngine)

    with pytest.raises(ValueError, match="unique"):
        engine.choice(
            state="test", question="test", candidates=["x", "x"]
        )


def test_choice_rejects_invalid_scoring_method():
    engine = DecisionEngine.__new__(DecisionEngine)

    with pytest.raises(ValueError, match="scoring"):
        engine.choice(
            state="test",
            question="test",
            candidates=["x", "y"],
            scoring="invalid",
        )


def test_choice_rejects_invalid_execution_mode():
    engine = DecisionEngine.__new__(DecisionEngine)

    with pytest.raises(ValueError, match="execution"):
        engine.choice(
            state="test",
            question="test",
            candidates=["x", "y"],
            execution="invalid",
        )


def test_sequential_execution_uses_one_forward_per_candidate():
    engine = make_engine()

    result = engine.choice(
        state="state",
        question="question",
        candidates=["alpha", "beta"],
        execution="sequential",
    )

    assert engine.model.calls == 2
    assert result.execution_mode == "sequential"
    assert result.selected == "alpha"


def test_batch_execution_uses_one_model_forward():
    engine = make_engine()

    result = engine.choice(
        state="state",
        question="question",
        candidates=["alpha", "beta"],
        execution="batch",
    )

    assert engine.model.calls == 1
    assert result.execution_mode == "batch"
    assert result.selected == "alpha"


def test_sequential_and_batch_match_in_float32_reference():
    sequential_engine = make_engine()
    batch_engine = make_engine()

    sequential = sequential_engine.choice(
        state="state",
        question="question",
        candidates=["alpha", "beta"],
        execution="sequential",
    )
    batched = batch_engine.choice(
        state="state",
        question="question",
        candidates=["alpha", "beta"],
        execution="batch",
    )

    assert sequential.scores == pytest.approx(batched.scores, abs=1e-6)
    assert sequential.probabilities == pytest.approx(
        batched.probabilities,
        abs=1e-6,
    )
    assert sequential.token_counts == batched.token_counts
    assert sequential.selected == batched.selected
