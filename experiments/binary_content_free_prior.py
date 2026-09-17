from __future__ import annotations

import gc
import json
import math
import statistics
from pathlib import Path

import torch

from llm_decision_engine import DecisionEngine


ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT
    / "results"
    / "binary_reasoning_model_size_v2_calibration.json"
)

OUTPUT = (
    ROOT
    / "results"
    / "binary_content_free_prior_calibration.json"
)

MODEL_IDS = (
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
)

PROBES = (
    {
        "name": "empty",
        "state": "",
        "question": "",
    },
    {
        "name": "na",
        "state": "N/A",
        "question": "N/A",
    },
    {
        "name": "mask",
        "state": "[MASK]",
        "question": "[MASK]",
    },
    {
        "name": "x",
        "state": "X",
        "question": "X",
    },
    {
        "name": "ellipsis",
        "state": "...",
        "question": "...",
    },
)


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)

    z = math.exp(x)
    return z / (1.0 + z)


def metrics(
    deltas: list[float],
    targets: list[bool],
    bias: float,
) -> dict:
    probabilities = [
        sigmoid(delta - bias)
        for delta in deltas
    ]

    predictions = [
        p >= 0.5
        for p in probabilities
    ]

    accuracy = sum(
        prediction == target
        for prediction, target
        in zip(predictions, targets)
    ) / len(targets)

    brier = statistics.mean(
        (
            p - (1.0 if target else 0.0)
        ) ** 2
        for p, target
        in zip(probabilities, targets)
    )

    eps = 1e-12

    nll = statistics.mean(
        -math.log(
            min(
                max(
                    p if target else 1.0 - p,
                    eps,
                ),
                1.0,
            )
        )
        for p, target
        in zip(probabilities, targets)
    )

    ece = 0.0
    num_bins = 10

    for bin_index in range(num_bins):
        lower = bin_index / num_bins
        upper = (
            bin_index + 1
        ) / num_bins

        indices = [
            i
            for i, p in enumerate(probabilities)
            if (
                p >= lower
                and (
                    p < upper
                    or (
                        bin_index == num_bins - 1
                        and p <= upper
                    )
                )
            )
        ]

        if not indices:
            continue

        mean_p = statistics.mean(
            probabilities[i]
            for i in indices
        )

        observed = statistics.mean(
            1.0 if targets[i] else 0.0
            for i in indices
        )

        ece += (
            len(indices)
            / len(probabilities)
            * abs(mean_p - observed)
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


def main() -> None:
    payload = json.loads(
        INPUT.read_text(
            encoding="utf-8"
        )
    )

    model_results = {
        model["model"]: model
        for model in payload["models"]
    }

    output_models = []

    for model_id in MODEL_IDS:
        print()
        print("=" * 80)
        print(model_id)
        print("=" * 80)

        engine = DecisionEngine(
            model=model_id
        )

        probe_results = []

        print()
        print("--- CONTENT-FREE PROBES ---")

        for probe in PROBES:
            result = engine.boolean(
                state=probe["state"],
                question=probe["question"],
                scoring="sum",
                execution="sequential",
            )

            bias = (
                result.scores["yes"]
                - result.scores["no"]
            )

            probe_results.append(
                {
                    "name": probe["name"],
                    "state": probe["state"],
                    "question": probe["question"],
                    "score_true": (
                        result.scores["yes"]
                    ),
                    "score_false": (
                        result.scores["no"]
                    ),
                    "bias": bias,
                    "probability_true": (
                        result.probability_true
                    ),
                }
            )

            print(
                f'{probe["name"]:<12}'
                f'b={bias:+10.6f}  '
                f'P(True)='
                f'{result.probability_true:.8f}'
            )

        biases = [
            row["bias"]
            for row in probe_results
        ]

        mean_bias = statistics.mean(
            biases
        )

        median_bias = statistics.median(
            biases
        )

        bias_stdev = statistics.stdev(
            biases
        )

        source = model_results[
            model_id
        ]

        rows = source["examples"]

        deltas = [
            float(
                row["direct"]["score_delta"]
            )
            for row in rows
        ]

        targets = [
            bool(row["target"])
            for row in rows
        ]

        raw = metrics(
            deltas,
            targets,
            bias=0.0,
        )

        individual_metrics = []

        for probe in probe_results:
            result = metrics(
                deltas,
                targets,
                bias=probe["bias"],
            )

            individual_metrics.append(
                {
                    "probe": probe["name"],
                    "bias": probe["bias"],
                    **result,
                }
            )

        mean_result = metrics(
            deltas,
            targets,
            bias=mean_bias,
        )

        median_result = metrics(
            deltas,
            targets,
            bias=median_bias,
        )

        print()
        print("--- PRIOR STABILITY ---")
        print(
            "Mean bias:   ",
            f"{mean_bias:+.6f}",
        )
        print(
            "Median bias: ",
            f"{median_bias:+.6f}",
        )
        print(
            "Bias stdev:  ",
            f"{bias_stdev:.6f}",
        )

        print()
        print("--- CALIBRATION RESULTS ---")
        print(
            f'{"Method":<20}'
            f'{"Accuracy":>10}'
            f'{"Brier":>10}'
            f'{"NLL":>10}'
            f'{"ECE":>10}'
            f'{"True rate":>12}'
        )

        def show(
            name: str,
            result: dict,
        ) -> None:
            print(
                f'{name:<20}'
                f'{result["accuracy"]:>10.3f}'
                f'{result["brier"]:>10.3f}'
                f'{result["nll"]:>10.3f}'
                f'{result["ece"]:>10.3f}'
                f'{result["predicted_true_rate"]:>12.3f}'
            )

        show(
            "raw",
            raw,
        )

        for result in individual_metrics:
            show(
                result["probe"],
                result,
            )

        show(
            "mean_prior",
            mean_result,
        )

        show(
            "median_prior",
            median_result,
        )

        output_models.append(
            {
                "model": model_id,
                "probes": probe_results,
                "bias_mean": mean_bias,
                "bias_median": median_bias,
                "bias_stdev": bias_stdev,
                "raw": raw,
                "probe_metrics": (
                    individual_metrics
                ),
                "mean_prior": mean_result,
                "median_prior": (
                    median_result
                ),
            }
        )

        del engine
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output_payload = {
        "evaluation_type": (
            "calibration_only_content_free_prior"
        ),
        "test_split_used": False,
        "source_results": str(
            INPUT.relative_to(ROOT)
        ),
        "generated_output_tokens": 0,
        "uses_target_labels_to_fit_bias": False,
        "models": output_models,
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
    print("FINAL COMPARISON")
    print("=" * 80)

    print(
        f'{"Model":<34}'
        f'{"Raw acc":>10}'
        f'{"Prior acc":>11}'
        f'{"Raw NLL":>10}'
        f'{"Prior NLL":>11}'
        f'{"Bias":>11}'
        f'{"Std":>9}'
    )

    for model in output_models:
        print(
            f'{model["model"]:<34}'
            f'{model["raw"]["accuracy"]:>10.3f}'
            f'{model["median_prior"]["accuracy"]:>11.3f}'
            f'{model["raw"]["nll"]:>10.3f}'
            f'{model["median_prior"]["nll"]:>11.3f}'
            f'{model["bias_median"]:>+11.3f}'
            f'{model["bias_stdev"]:>9.3f}'
        )

    print()
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()
