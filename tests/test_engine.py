import pytest

from llm_decision_engine import DecisionEngine


def test_choice_requires_at_least_two_candidates():
    engine = DecisionEngine.__new__(DecisionEngine)

    with pytest.raises(ValueError, match="at least two candidates"):
        engine.choice(
            state="test state",
            question="test question",
            candidates=["only one"],
        )


def test_single_token_baseline_rejects_more_than_26_candidates():
    engine = DecisionEngine.__new__(DecisionEngine)

    with pytest.raises(ValueError, match="at most 26 candidates"):
        engine.choice(
            state="test state",
            question="test question",
            candidates=[f"candidate {i}" for i in range(27)],
        )
