import math
from collections.abc import Sequence

from .types import BinaryCalibrationResult, BinaryEvaluationResult, CalibrationBin


def binary_brier_score(probability_true: float, target: bool) -> float:
    if not isinstance(target, bool):
        raise TypeError("target must be a boolean.")

    if not 0.0 <= probability_true <= 1.0:
        raise ValueError("probability_true must be between 0 and 1.")

    target_value = 1.0 if target else 0.0
    return (probability_true - target_value) ** 2


def negative_log_likelihood(probability: float) -> float:
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between 0 and 1.")

    if probability == 0.0:
        return math.inf

    return -math.log(probability)


def evaluate_binary_predictions(
    probabilities_true: Sequence[float],
    targets: Sequence[bool],
    *,
    threshold: float = 0.5,
) -> BinaryEvaluationResult:
    if len(probabilities_true) != len(targets):
        raise ValueError("probabilities_true and targets must have the same length.")

    if not probabilities_true:
        raise ValueError("At least one prediction is required.")

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1.")

    brier_scores = []
    nll_values = []
    correct = 0

    for probability_true, target in zip(probabilities_true, targets):
        brier_scores.append(binary_brier_score(probability_true, target))

        correct_probability = (
            probability_true
            if target
            else 1.0 - probability_true
        )
        nll_values.append(negative_log_likelihood(correct_probability))

        predicted = probability_true >= threshold
        if predicted == target:
            correct += 1

    count = len(probabilities_true)

    return BinaryEvaluationResult(
        count=count,
        accuracy=correct / count,
        mean_brier=math.fsum(brier_scores) / count,
        mean_nll=math.fsum(nll_values) / count,
    )

def evaluate_binary_calibration(
    probabilities_true: Sequence[float],
    targets: Sequence[bool],
    *,
    num_bins: int = 10,
) -> BinaryCalibrationResult:
    if len(probabilities_true) != len(targets):
        raise ValueError("probabilities_true and targets must have the same length.")

    if not probabilities_true:
        raise ValueError("At least one prediction is required.")

    if isinstance(num_bins, bool) or not isinstance(num_bins, int):
        raise TypeError("num_bins must be an integer.")

    if num_bins <= 0:
        raise ValueError("num_bins must be greater than zero.")

    counts = [0] * num_bins
    probability_sums = [0.0] * num_bins
    target_sums = [0] * num_bins

    for probability_true, target in zip(probabilities_true, targets):
        binary_brier_score(probability_true, target)
        bin_index = min(int(probability_true * num_bins), num_bins - 1)
        counts[bin_index] += 1
        probability_sums[bin_index] += probability_true
        target_sums[bin_index] += 1 if target else 0

    total = len(probabilities_true)
    bins = []
    weighted_gaps = []

    for bin_index, count in enumerate(counts):
        if count == 0:
            continue

        mean_probability_true = probability_sums[bin_index] / count
        observed_true_rate = target_sums[bin_index] / count
        absolute_gap = abs(mean_probability_true - observed_true_rate)

        bins.append(
            CalibrationBin(
                lower_bound=bin_index / num_bins,
                upper_bound=(bin_index + 1) / num_bins,
                count=count,
                mean_probability_true=mean_probability_true,
                observed_true_rate=observed_true_rate,
                absolute_gap=absolute_gap,
            )
        )
        weighted_gaps.append((count / total) * absolute_gap)

    return BinaryCalibrationResult(
        count=total,
        num_bins=num_bins,
        expected_calibration_error=math.fsum(weighted_gaps),
        bins=tuple(bins),
    )
