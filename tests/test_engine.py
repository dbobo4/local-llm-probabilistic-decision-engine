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


def test_choice_rejects_duplicate_candidates():
    engine = DecisionEngine.__new__(DecisionEngine)

    with pytest.raises(ValueError, match="unique"):
        engine.choice(
            state="test state",
            question="test question",
            candidates=["billing", "billing"],
        )


def test_choice_rejects_invalid_scoring_method():
    engine = DecisionEngine.__new__(DecisionEngine)

    with pytest.raises(ValueError, match="scoring"):
        engine.choice(
            state="test state",
            question="test question",
            candidates=["billing", "sales"],
            scoring="invalid",
        )
