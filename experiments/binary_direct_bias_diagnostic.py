from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT
    / "results"
    / "binary_reasoning_model_size_v2_calibration.json"
)

OUTPUT = (
    ROOT
    / "results"
    / "binary_direct_bias_diagnostic_calibration.json"
)

EXPECTED_DATASET_SHA256 = (
    "d2116a228b48c959f3c77e8e4ecfa816"
    "033aa4044ec55bbfb930553985a2e4b8"
)

NUM_FOLDS = 5


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)

    z = math.exp(x)
    return z / (1.0 + z)


def safe_logit(p: float) -> float:
    eps = 1e-12
    p = min(max(p, eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


def extract_delta_and_probability(
    direct: dict,
) -> tuple[float, float]:
    probability_true = None

    if "probability_true" in direct:
        probability_true = float(
            direct["probability_true"]
        )

    elif "probabilities" in direct:
        probs = direct["probabilities"]

        if "yes" in probs:
            probability_true = float(
                probs["yes"]
            )
        elif "True" in probs:
            probability_true = float(
                probs["True"]
            )

    delta = None

    if "score_delta" in direct:
        delta = float(
            direct["score_delta"]
        )

    elif (
        "score_true" in direct
        and "score_false" in direct
    ):
        delta = (
            float(direct["score_true"])
            - float(direct["score_false"])
        )

    elif "scores" in direct:
        scores = direct["scores"]

        if "yes" in scores and "no" in scores:
            delta = (
                float(scores["yes"])
                - float(scores["no"])
            )
        elif (
            "True" in scores
            and "False" in scores
        ):
            delta = (
                float(scores["True"])
                - float(scores["False"])
            )

    if delta is None and probability_true is not None:
        delta = safe_logit(
            probability_true
        )

    if probability_true is None and delta is not None:
        probability_true = sigmoid(
            delta
        )

    if delta is None or probability_true is None:
        raise RuntimeError(
            "Could not extract direct score delta "
            "and probability_true."
        )

    return delta, probability_true


def metrics(
    probabilities: list[float],
    targets: list[bool],
) -> dict:
    eps = 1e-12

    predictions = [
        probability >= 0.5
        for probability in probabilities
    ]

    accuracy = sum(
        prediction == target
        for prediction, target
        in zip(predictions, targets)
    ) / len(targets)

    brier = statistics.mean(
        (
            probability
            - (1.0 if target else 0.0)
        ) ** 2
        for probability, target
        in zip(probabilities, targets)
    )

    nll_terms = []

    for probability, target in zip(
        probabilities,
        targets,
    ):
        probability = min(
            max(probability, eps),
            1.0 - eps,
        )

        if target:
            nll_terms.append(
                -math.log(probability)
            )
        else:
            nll_terms.append(
                -math.log(1.0 - probability)
            )

    nll = statistics.mean(
        nll_terms
    )

    ece = 0.0
    num_bins = 10

    for bin_index in range(num_bins):
        lower = bin_index / num_bins
        upper = (
            (bin_index + 1)
            / num_bins
        )

        indices = [
            i
            for i, probability
            in enumerate(probabilities)
            if (
                probability >= lower
                and (
                    probability < upper
                    or (
                        bin_index
                        == num_bins - 1
                        and probability <= upper
                    )
                )
            )
        ]

        if not indices:
            continue

        mean_probability = statistics.mean(
            probabilities[i]
            for i in indices
        )

        observed_rate = statistics.mean(
            1.0 if targets[i] else 0.0
            for i in indices
        )

        ece += (
            len(indices)
            / len(probabilities)
            * abs(
                mean_probability
                - observed_rate
            )
        )

    return {
        "accuracy": accuracy,
        "brier": brier,
        "nll": nll,
        "ece": ece,
        "predicted_true_rate": (
            sum(predictions)
            / len(predictions)
        ),
    }


def fit_nll_bias(
    deltas: list[float],
    targets: list[bool],
) -> float:
    target_rate = statistics.mean(
        1.0 if target else 0.0
        for target in targets
    )

    low = min(deltas) - 50.0
    high = max(deltas) + 50.0

    for _ in range(200):
        mid = (
            low + high
        ) / 2.0

        predicted_rate = statistics.mean(
            sigmoid(delta - mid)
            for delta in deltas
        )

        if predicted_rate > target_rate:
            low = mid
        else:
            high = mid

    return (
        low + high
    ) / 2.0


def fit_accuracy_threshold(
    deltas: list[float],
    targets: list[bool],
) -> float:
    unique = sorted(
        set(deltas)
    )

    candidates = [
        unique[0] - 1.0
    ]

    candidates.extend(
        (
            left + right
        ) / 2.0
        for left, right
        in zip(
            unique,
            unique[1:],
        )
    )

    candidates.append(
        unique[-1] + 1.0
    )

    best_accuracy = -1.0
    best_thresholds = []

    for threshold in candidates:
        current_accuracy = sum(
            (
                delta >= threshold
            ) == target
            for delta, target
            in zip(
                deltas,
                targets,
            )
        ) / len(targets)

        if current_accuracy > best_accuracy:
            best_accuracy = (
                current_accuracy
            )
            best_thresholds = [
                threshold
            ]

        elif current_accuracy == best_accuracy:
            best_thresholds.append(
                threshold
            )

    return min(
        best_thresholds,
        key=abs,
    )


def assign_folds(
    rows: list[dict],
) -> dict[str, int]:
    strata = defaultdict(list)

    for row in rows:
        key = (
            row["category"],
            row["target"],
        )

        strata[key].append(
            row
        )

    fold_by_id = {}

    for stratum_rows in strata.values():
        stratum_rows = sorted(
            stratum_rows,
            key=lambda row: row["id"],
        )

        for index, row in enumerate(
            stratum_rows
        ):
            fold_by_id[row["id"]] = (
                index % NUM_FOLDS
            )

    return fold_by_id


def category_accuracy(
    rows: list[dict],
    prediction_field: str,
) -> dict[str, float]:
    output = {}

    categories = sorted(
        {
            row["category"]
            for row in rows
        }
    )

    for category in categories:
        subset = [
            row
            for row in rows
            if row["category"] == category
        ]

        output[category] = sum(
            row[prediction_field]
            == row["target"]
            for row in subset
        ) / len(subset)

    return output


def analyze_model(
    model: dict,
) -> dict:
    model_id = model["model"]

    rows = []

    for example in model["examples"]:
        delta, probability_true = (
            extract_delta_and_probability(
                example["direct"]
            )
        )

        rows.append(
            {
                "id": example["id"],
                "category": (
                    example["category"]
                ),
                "target": bool(
                    example["target"]
                ),
                "delta": delta,
                "raw_probability_true": (
                    probability_true
                ),
            }
        )

    fold_by_id = assign_folds(
        rows
    )

    true_deltas = [
        row["delta"]
        for row in rows
        if row["target"]
    ]

    false_deltas = [
        row["delta"]
        for row in rows
        if not row["target"]
    ]

    full_nll_bias = fit_nll_bias(
        [row["delta"] for row in rows],
        [row["target"] for row in rows],
    )

    full_accuracy_threshold = (
        fit_accuracy_threshold(
            [
                row["delta"]
                for row in rows
            ],
            [
                row["target"]
                for row in rows
            ],
        )
    )

    cv_rows = []
    fold_summaries = []

    for fold in range(NUM_FOLDS):
        train = [
            row
            for row in rows
            if fold_by_id[row["id"]]
            != fold
        ]

        validation = [
            row
            for row in rows
            if fold_by_id[row["id"]]
            == fold
        ]

        train_deltas = [
            row["delta"]
            for row in train
        ]

        train_targets = [
            row["target"]
            for row in train
        ]

        nll_bias = fit_nll_bias(
            train_deltas,
            train_targets,
        )

        accuracy_threshold = (
            fit_accuracy_threshold(
                train_deltas,
                train_targets,
            )
        )

        fold_summaries.append(
            {
                "fold": fold,
                "train_count": len(train),
                "validation_count": (
                    len(validation)
                ),
                "nll_bias": nll_bias,
                "accuracy_threshold": (
                    accuracy_threshold
                ),
            }
        )

        for row in validation:
            raw_probability = (
                row[
                    "raw_probability_true"
                ]
            )

            nll_probability = sigmoid(
                row["delta"]
                - nll_bias
            )

            accuracy_probability = sigmoid(
                row["delta"]
                - accuracy_threshold
            )

            cv_rows.append(
                {
                    **row,
                    "fold": fold,
                    "raw_probability": (
                        raw_probability
                    ),
                    "nll_corrected_probability": (
                        nll_probability
                    ),
                    "accuracy_corrected_probability": (
                        accuracy_probability
                    ),
                    "raw_prediction": (
                        raw_probability >= 0.5
                    ),
                    "nll_corrected_prediction": (
                        nll_probability >= 0.5
                    ),
                    "accuracy_corrected_prediction": (
                        accuracy_probability >= 0.5
                    ),
                }
            )

    cv_rows = sorted(
        cv_rows,
        key=lambda row: row["id"],
    )

    targets = [
        row["target"]
        for row in cv_rows
    ]

    raw_metrics = metrics(
        [
            row["raw_probability"]
            for row in cv_rows
        ],
        targets,
    )

    nll_metrics = metrics(
        [
            row[
                "nll_corrected_probability"
            ]
            for row in cv_rows
        ],
        targets,
    )

    accuracy_metrics = metrics(
        [
            row[
                "accuracy_corrected_probability"
            ]
            for row in cv_rows
        ],
        targets,
    )

    categories = {
        "raw": category_accuracy(
            cv_rows,
            "raw_prediction",
        ),
        "nll_intercept": (
            category_accuracy(
                cv_rows,
                "nll_corrected_prediction",
            )
        ),
        "accuracy_threshold": (
            category_accuracy(
                cv_rows,
                "accuracy_corrected_prediction",
            )
        ),
    }

    print()
    print("=" * 80)
    print(model_id)
    print("=" * 80)

    print()
    print("--- SCORE DELTA DISTRIBUTION ---")
    print(
        "True target mean delta:   ",
        f"{statistics.mean(true_deltas):+.6f}",
    )
    print(
        "True target median delta: ",
        f"{statistics.median(true_deltas):+.6f}",
    )
    print(
        "False target mean delta:  ",
        f"{statistics.mean(false_deltas):+.6f}",
    )
    print(
        "False target median delta:",
        f"{statistics.median(false_deltas):+.6f}",
    )

    print()
    print("--- FULL CALIBRATION FIT ---")
    print(
        "NLL-optimal bias b:       ",
        f"{full_nll_bias:+.6f}",
    )
    print(
        "Accuracy threshold b:     ",
        f"{full_accuracy_threshold:+.6f}",
    )

    print()
    print("--- 5-FOLD CROSS-VALIDATED RESULTS ---")
    print(
        f'{"Method":<24}'
        f'{"Accuracy":>10}'
        f'{"Brier":>10}'
        f'{"NLL":>10}'
        f'{"ECE":>10}'
        f'{"True rate":>12}'
    )

    for name, result in (
        ("raw", raw_metrics),
        ("nll_intercept", nll_metrics),
        (
            "accuracy_threshold",
            accuracy_metrics,
        ),
    ):
        print(
            f'{name:<24}'
            f'{result["accuracy"]:>10.3f}'
            f'{result["brier"]:>10.3f}'
            f'{result["nll"]:>10.3f}'
            f'{result["ece"]:>10.3f}'
            f'{result["predicted_true_rate"]:>12.3f}'
        )

    print()
    print("--- FOLD PARAMETERS ---")

    for fold in fold_summaries:
        print(
            f'fold {fold["fold"]}:  '
            f'NLL b={fold["nll_bias"]:+.4f}  '
            f'ACC b={fold["accuracy_threshold"]:+.4f}'
        )

    print()
    print("--- CATEGORY ACCURACY ---")
    print(
        f'{"Category":<16}'
        f'{"Raw":>10}'
        f'{"NLL-bias":>12}'
        f'{"ACC-thresh":>12}'
    )

    for category in sorted(
        categories["raw"]
    ):
        print(
            f'{category:<16}'
            f'{categories["raw"][category]:>10.3f}'
            f'{categories["nll_intercept"][category]:>12.3f}'
            f'{categories["accuracy_threshold"][category]:>12.3f}'
        )

    return {
        "model": model_id,
        "example_count": len(rows),
        "score_delta_distribution": {
            "true_mean": statistics.mean(
                true_deltas
            ),
            "true_median": statistics.median(
                true_deltas
            ),
            "false_mean": statistics.mean(
                false_deltas
            ),
            "false_median": statistics.median(
                false_deltas
            ),
        },
        "full_calibration_fit": {
            "nll_bias": full_nll_bias,
            "accuracy_threshold": (
                full_accuracy_threshold
            ),
        },
        "cross_validation": {
            "folds": fold_summaries,
            "raw": raw_metrics,
            "nll_intercept": nll_metrics,
            "accuracy_threshold": (
                accuracy_metrics
            ),
            "categories": categories,
        },
        "examples": cv_rows,
    }


def main() -> None:
    payload = json.loads(
        INPUT.read_text(
            encoding="utf-8"
        )
    )

    dataset_hash = payload.get(
        "dataset_sha256"
    )

    if (
        dataset_hash is not None
        and dataset_hash.lower()
        != EXPECTED_DATASET_SHA256
    ):
        raise RuntimeError(
            "Calibration dataset hash mismatch."
        )

    if payload.get("split") not in (
        None,
        "calibration",
    ):
        raise RuntimeError(
            "Expected calibration-only source results."
        )

    models = payload["models"]

    results = [
        analyze_model(model)
        for model in models
    ]

    output_payload = {
        "evaluation_type": (
            "calibration_only_direct_bias_diagnostic"
        ),
        "test_split_used": False,
        "source_results": str(
            INPUT.relative_to(ROOT)
        ),
        "dataset_sha256": (
            EXPECTED_DATASET_SHA256
        ),
        "num_folds": NUM_FOLDS,
        "decision_rule": (
            "True iff score_true - score_false >= threshold"
        ),
        "results": results,
    }

    OUTPUT.write_text(
        json.dumps(
            output_payload,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(
        f'{"Model":<34}'
        f'{"Raw":>9}'
        f'{"CV NLL":>10}'
        f'{"CV ACC":>10}'
        f'{"Full NLL b":>13}'
        f'{"Full ACC b":>13}'
    )

    for result in results:
        cv = result[
            "cross_validation"
        ]

        fit = result[
            "full_calibration_fit"
        ]

        print(
            f'{result["model"]:<34}'
            f'{cv["raw"]["accuracy"]:>9.3f}'
            f'{cv["nll_intercept"]["accuracy"]:>10.3f}'
            f'{cv["accuracy_threshold"]["accuracy"]:>10.3f}'
            f'{fit["nll_bias"]:>+13.4f}'
            f'{fit["accuracy_threshold"]:>+13.4f}'
        )

    print()
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()
