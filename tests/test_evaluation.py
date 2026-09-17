import math

import pytest

from llm_decision_engine.evaluation import (
    binary_brier_score,
    negative_log_likelihood,
)


def test_binary_brier_score_for_correct_confident_prediction():
    assert binary_brier_score(0.9, True) == pytest.approx(0.01)


def test_binary_brier_score_for_false_target():
    assert binary_brier_score(0.2, False) == pytest.approx(0.04)


def test_binary_brier_score_rejects_invalid_probability():
    with pytest.raises(ValueError, match="between 0 and 1"):
        binary_brier_score(1.1, True)


def test_negative_log_likelihood():
    assert negative_log_likelihood(0.9) == pytest.approx(-math.log(0.9))


def test_negative_log_likelihood_penalizes_low_correct_probability():
    assert negative_log_likelihood(0.1) > negative_log_likelihood(0.9)


def test_negative_log_likelihood_rejects_zero():
    with pytest.raises(ValueError, match="greater than 0"):
        negative_log_likelihood(0.0)

def test_binary_brier_score_rejects_non_boolean_target():
    with pytest.raises(TypeError, match="boolean"):
        binary_brier_score(0.9, 1)
