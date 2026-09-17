import math
from collections.abc import Sequence

from .types import BinaryEvaluationResult


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
