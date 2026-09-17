import math

import pytest

from llm_decision_engine.evaluation import (
    binary_brier_score,
    evaluate_binary_predictions,
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


def test_negative_log_likelihood_returns_infinity_for_zero():
    assert negative_log_likelihood(0.0) == math.inf

def test_binary_brier_score_rejects_non_boolean_target():
    with pytest.raises(TypeError, match="boolean"):
        binary_brier_score(0.9, 1)

def test_negative_log_likelihood_rejects_out_of_range_probability():
    with pytest.raises(ValueError, match="between 0 and 1"):
        negative_log_likelihood(1.1)

def test_evaluate_binary_predictions():
    probabilities = [0.9, 0.8, 0.3, 0.1]
    targets = [True, True, False, False]

    result = evaluate_binary_predictions(probabilities, targets)

    assert result.count == 4
    assert result.accuracy == pytest.approx(1.0)
    assert result.mean_brier == pytest.approx(0.0375)
    assert result.mean_nll == pytest.approx(
        sum(
            negative_log_likelihood(probability)
            for probability in [0.9, 0.8, 0.7, 0.9]
        ) / 4
    )


def test_evaluate_binary_predictions_uses_threshold():
    result = evaluate_binary_predictions(
        [0.6, 0.4],
        [False, False],
        threshold=0.7,
    )

    assert result.accuracy == pytest.approx(1.0)


def test_evaluate_binary_predictions_rejects_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        evaluate_binary_predictions([0.9], [True, False])


def test_evaluate_binary_predictions_rejects_empty_dataset():
    with pytest.raises(ValueError, match="At least one"):
        evaluate_binary_predictions([], [])


def test_evaluate_binary_predictions_rejects_invalid_threshold():
    with pytest.raises(ValueError, match="threshold"):
        evaluate_binary_predictions([0.9], [True], threshold=1.1)

def test_evaluate_binary_predictions_propagates_infinite_nll():
    result = evaluate_binary_predictions(
        [0.0, 0.9],
        [True, True],
    )

    assert result.count == 2
    assert result.accuracy == pytest.approx(0.5)
    assert result.mean_brier == pytest.approx(0.505)
    assert result.mean_nll == math.inf
