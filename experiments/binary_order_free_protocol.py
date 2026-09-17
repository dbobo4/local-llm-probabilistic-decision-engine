from __future__ import annotations

import gc
import hashlib
import json
import math
import statistics
from pathlib import Path

import torch

from llm_decision_engine import DecisionEngine


ROOT = Path(__file__).resolve().parents[1]

DATASET = (
    ROOT
    / "benchmarks"
    / "data"
    / "binary_quality_v2_calibration.json"
)

BASELINE_RESULTS = (
    ROOT
    / "results"
    / "binary_reasoning_model_size_v2_calibration.json"
)

OUTPUT = (
    ROOT
    / "results"
    / "binary_order_free_protocol_calibration.json"
)

EXPECTED_DATASET_SHA256 = (
    "d2116a228b48c959f3c77e8e4ecfa816"
    "033aa4044ec55bbfb930553985a2e4b8"
)

MODEL_IDS = (
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
)

PROTOCOLS = (
    (
        "boolean_literal",
        "Answer with exactly one Boolean literal.",
    ),
    (
        "boolean_value",
        "Return exactly one Boolean value.",
    ),
    (
        "single_boolean_literal",
        "Respond using a single Boolean literal and nothing else.",
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)

    z = math.exp(x)
    return z / (1.0 + z)


def build_prefix(
    tokenizer,
    state: str,
    question: str,
    instruction: str,
) -> str:
    content = (
        f"STATE:\n{state}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"{instruction}"
    )

    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )


def continuation_ids(
    tokenizer,
    prefix: str,
    continuation: str,
) -> tuple[list[int], list[int]]:
    prefix_ids = tokenizer(
        prefix,
        add_special_tokens=False,
    )["input_ids"]

    full_ids = tokenizer(
        prefix + continuation,
        add_special_tokens=False,
    )["input_ids"]

    if full_ids[:len(prefix_ids)] != prefix_ids:
        raise RuntimeError(
            "Tokenizer boundary mismatch."
        )

    target_ids = full_ids[len(prefix_ids):]

    if not target_ids:
        raise RuntimeError(
            "Empty continuation."
        )

    return full_ids, target_ids


def score_candidate(
    engine: DecisionEngine,
    prefix: str,
    candidate: str,
) -> dict:
    full_ids, target_ids = continuation_ids(
        engine.tokenizer,
        prefix,
        candidate,
    )

    prefix_length = (
        len(full_ids)
        - len(target_ids)
    )

    device = next(
        engine.model.parameters()
    ).device

    input_ids = torch.tensor(
        [full_ids],
        dtype=torch.long,
        device=device,
    )

    attention_mask = torch.ones_like(
        input_ids
    )

    with torch.inference_mode():
        logits = engine.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits[0]

    log_probs = torch.log_softmax(
        logits.float(),
        dim=-1,
    )

    start = prefix_length - 1

    score = 0.0

    for offset, token_id in enumerate(
        target_ids
    ):
        score += float(
            log_probs[
                start + offset,
                token_id,
            ].item()
        )

    first_token_probability = float(
        torch.exp(
            log_probs[
                start,
                target_ids[0],
            ]
        ).item()
    )

    return {
        "score": score,
        "token_count": len(target_ids),
        "first_token_probability": (
            first_token_probability
        ),
    }


def metrics(
    probabilities: list[float],
    targets: list[bool],
) -> dict:
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

    eps = 1e-12

    nll = statistics.mean(
        -math.log(
            max(
                probability
                if target
                else 1.0 - probability,
                eps,
            )
        )
        for probability, target
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
            for i, probability
            in enumerate(probabilities)
            if (
                probability >= lower
                and (
                    probability < upper
                    or (
                        bin_index == num_bins - 1
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


def main() -> None:
    dataset_hash = sha256_file(
        DATASET
    )

    if dataset_hash != EXPECTED_DATASET_SHA256:
        raise RuntimeError(
            "Calibration dataset hash mismatch."
        )

    examples = json.loads(
        DATASET.read_text(
            encoding="utf-8"
        )
    )

    baseline_payload = json.loads(
        BASELINE_RESULTS.read_text(
            encoding="utf-8"
        )
    )

    baseline_models = {
        model["model"]: model
        for model in baseline_payload["models"]
    }

    all_models = []

    for model_id in MODEL_IDS:
        print()
        print("=" * 80)
        print(model_id)
        print("=" * 80)

        engine = DecisionEngine(
            model=model_id
        )

        baseline = baseline_models[
            model_id
        ]

        print(
            "Dedicated baseline accuracy:",
            f'{baseline["direct_accuracy"]:.6f}',
        )

        protocol_results = []

        for protocol_name, instruction in PROTOCOLS:
            print()
            print("-" * 80)
            print(
                "PROTOCOL:",
                protocol_name,
            )
            print(
                "Instruction:",
                instruction,
            )
            print("-" * 80)

            rows = []

            for index, example in enumerate(
                examples,
                start=1,
            ):
                prefix = build_prefix(
                    engine.tokenizer,
                    example["state"],
                    example["question"],
                    instruction,
                )

                true_result = score_candidate(
                    engine,
                    prefix,
                    "True",
                )

                false_result = score_candidate(
                    engine,
                    prefix,
                    "False",
                )

                delta = (
                    true_result["score"]
                    - false_result["score"]
                )

                probability_true = sigmoid(
                    delta
                )

                selected = (
                    delta >= 0.0
                )

                candidate_mass = None

                if (
                    true_result["token_count"] == 1
                    and false_result["token_count"] == 1
                ):
                    candidate_mass = (
                        true_result[
                            "first_token_probability"
                        ]
                        + false_result[
                            "first_token_probability"
                        ]
                    )

                rows.append(
                    {
                        "id": example["id"],
                        "category": (
                            example["category"]
                        ),
                        "target": (
                            example["target"]
                        ),
                        "score_true": (
                            true_result["score"]
                        ),
                        "score_false": (
                            false_result["score"]
                        ),
                        "delta": delta,
                        "probability_true": (
                            probability_true
                        ),
                        "selected": selected,
                        "correct": (
                            selected
                            == example["target"]
                        ),
                        "true_token_count": (
                            true_result[
                                "token_count"
                            ]
                        ),
                        "false_token_count": (
                            false_result[
                                "token_count"
                            ]
                        ),
                        "candidate_mass": (
                            candidate_mass
                        ),
                    }
                )

                print(
                    f"{index:3d}/120  "
                    f'{example["id"]:<40} '
                    f'target={str(example["target"]):5}  '
                    f'selected={str(selected):5}  '
                    f'delta={delta:+8.3f}'
                )

            targets = [
                bool(row["target"])
                for row in rows
            ]

            probabilities = [
                row["probability_true"]
                for row in rows
            ]

            result_metrics = metrics(
                probabilities,
                targets,
            )

            masses = [
                row["candidate_mass"]
                for row in rows
                if row["candidate_mass"]
                is not None
            ]

            categories = []

            for category in sorted(
                {
                    row["category"]
                    for row in rows
                }
            ):
                subset = [
                    row
                    for row in rows
                    if row["category"]
                    == category
                ]

                categories.append(
                    {
                        "category": category,
                        "accuracy": (
                            sum(
                                row["correct"]
                                for row in subset
                            )
                            / len(subset)
                        ),
                    }
                )

            protocol_result = {
                "protocol": (
                    protocol_name
                ),
                "instruction": (
                    instruction
                ),
                "metrics": (
                    result_metrics
                ),
                "candidate_mass_mean": (
                    statistics.mean(masses)
                    if masses
                    else None
                ),
                "candidate_mass_median": (
                    statistics.median(masses)
                    if masses
                    else None
                ),
                "candidate_mass_min": (
                    min(masses)
                    if masses
                    else None
                ),
                "single_token_candidates": (
                    len(masses)
                    == len(rows)
                ),
                "categories": categories,
                "examples": rows,
            }

            protocol_results.append(
                protocol_result
            )

            print()
            print("--- SUMMARY ---")
            print(
                "Accuracy:       ",
                f'{result_metrics["accuracy"]:.6f}',
            )
            print(
                "Brier:          ",
                f'{result_metrics["brier"]:.6f}',
            )
            print(
                "NLL:            ",
                f'{result_metrics["nll"]:.6f}',
            )
            print(
                "ECE:            ",
                f'{result_metrics["ece"]:.6f}',
            )
            print(
                "True rate:      ",
                f'{result_metrics["predicted_true_rate"]:.6f}',
            )

            if masses:
                print(
                    "Candidate mass mean:  ",
                    f"{statistics.mean(masses):.6f}",
                )
                print(
                    "Candidate mass median:",
                    f"{statistics.median(masses):.6f}",
                )
                print(
                    "Candidate mass min:   ",
                    f"{min(masses):.6f}",
                )

            print()
            print("By category:")

            for row in categories:
                print(
                    f'  {row["category"]:<16}'
                    f'{row["accuracy"]:.3f}'
                )

        all_models.append(
            {
                "model": model_id,
                "dedicated_baseline_accuracy": (
                    baseline[
                        "direct_accuracy"
                    ]
                ),
                "protocols": (
                    protocol_results
                ),
            }
        )

        del engine
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output_payload = {
        "evaluation_type": (
            "calibration_only_order_free_binary_protocol"
        ),
        "test_split_used": False,
        "dataset": str(
            DATASET.relative_to(ROOT)
        ),
        "dataset_sha256": (
            dataset_hash
        ),
        "generated_output_tokens": 0,
        "uses_reasoning": False,
        "uses_generate": False,
        "models": all_models,
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
        f'{"Baseline":>10}'
        f'{"Literal":>10}'
        f'{"Value":>10}'
        f'{"Single":>10}'
    )

    for model in all_models:
        protocols = {
            row["protocol"]: row
            for row in model["protocols"]
        }

        print(
            f'{model["model"]:<34}'
            f'{model["dedicated_baseline_accuracy"]:>10.3f}'
            f'{protocols["boolean_literal"]["metrics"]["accuracy"]:>10.3f}'
            f'{protocols["boolean_value"]["metrics"]["accuracy"]:>10.3f}'
            f'{protocols["single_boolean_literal"]["metrics"]["accuracy"]:>10.3f}'
        )

    print()
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()
