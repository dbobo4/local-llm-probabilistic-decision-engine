import math

import pytest

from llm_decision_engine.evaluation import (
    binary_brier_score,
    evaluate_binary_calibration,
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

def test_evaluate_binary_calibration_basic():
    result = evaluate_binary_calibration(
        [0.1, 0.2, 0.8, 0.9],
        [False, False, True, False],
        num_bins=2,
    )

    assert result.count == 4
    assert result.num_bins == 2
    assert result.expected_calibration_error == pytest.approx(0.25)
    assert len(result.bins) == 2
    assert result.bins[0].count == 2
    assert result.bins[0].mean_probability_true == pytest.approx(0.15)
    assert result.bins[0].observed_true_rate == pytest.approx(0.0)
    assert result.bins[1].count == 2
    assert result.bins[1].mean_probability_true == pytest.approx(0.85)
    assert result.bins[1].observed_true_rate == pytest.approx(0.5)


def test_evaluate_binary_calibration_places_one_in_last_bin():
    result = evaluate_binary_calibration([1.0], [True], num_bins=10)
    assert len(result.bins) == 1
    assert result.bins[0].lower_bound == pytest.approx(0.9)
    assert result.bins[0].upper_bound == pytest.approx(1.0)


def test_evaluate_binary_calibration_rejects_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        evaluate_binary_calibration([0.8], [True, False])


def test_evaluate_binary_calibration_rejects_empty_dataset():
    with pytest.raises(ValueError, match="At least one"):
        evaluate_binary_calibration([], [])


def test_evaluate_binary_calibration_rejects_invalid_bin_count():
    with pytest.raises(ValueError, match="greater than zero"):
        evaluate_binary_calibration([0.8], [True], num_bins=0)


def test_evaluate_binary_calibration_rejects_boolean_bin_count():
    with pytest.raises(TypeError, match="integer"):
        evaluate_binary_calibration([0.8], [True], num_bins=True)

def test_evaluate_binary_calibration_bin_boundaries():
    result = evaluate_binary_calibration(
        [0.0, 0.5, 1.0],
        [False, True, True],
        num_bins=2,
    )

    assert len(result.bins) == 2

    first, second = result.bins

    assert first.lower_bound == pytest.approx(0.0)
    assert first.upper_bound == pytest.approx(0.5)
    assert first.count == 1

    assert second.lower_bound == pytest.approx(0.5)
    assert second.upper_bound == pytest.approx(1.0)
    assert second.count == 2


def test_binary_brier_score_rejects_non_real_probability():
    for probability in (True, "0.5", None):
        with pytest.raises(TypeError, match="real number"):
            binary_brier_score(probability, True)


def test_negative_log_likelihood_rejects_non_real_probability():
    for probability in (True, "0.5", None):
        with pytest.raises(TypeError, match="real number"):
            negative_log_likelihood(probability)


def test_probability_metrics_reject_non_finite_values():
    for probability in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="between 0 and 1"):
            binary_brier_score(probability, True)

        with pytest.raises(ValueError, match="between 0 and 1"):
            negative_log_likelihood(probability)


def test_evaluate_binary_predictions_rejects_invalid_threshold_type():
    for threshold in (True, "0.5", None):
        with pytest.raises(TypeError, match="real number"):
            evaluate_binary_predictions(
                [0.9],
                [True],
                threshold=threshold,
            )


def test_evaluate_binary_predictions_rejects_non_finite_threshold():
    for threshold in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="between 0 and 1"):
            evaluate_binary_predictions(
                [0.9],
                [True],
                threshold=threshold,
            )


def test_evaluate_binary_calibration_rejects_invalid_probability_type():
    with pytest.raises(TypeError, match="real number"):
        evaluate_binary_calibration(
            ["0.9"],
            [True],
        )
