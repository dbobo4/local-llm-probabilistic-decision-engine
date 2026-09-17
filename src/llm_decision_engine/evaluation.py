import math


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
