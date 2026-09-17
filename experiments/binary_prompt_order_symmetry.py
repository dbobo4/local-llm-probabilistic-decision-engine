from __future__ import annotations

import gc
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

OUTPUT = (
    ROOT
    / "results"
    / "binary_prompt_order_symmetry_calibration.json"
)

MODEL_IDS = (
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
)


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)

    z = math.exp(x)
    return z / (1.0 + z)


def make_prefix(
    tokenizer,
    state: str,
    question: str,
    reversed_order: bool,
) -> str:
    if reversed_order:
        instruction = (
            "Answer exactly one word: False or True."
        )
    else:
        instruction = (
            "Answer exactly one word: True or False."
        )

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

    target_ids = full_ids[
        len(prefix_ids):
    ]

    if not target_ids:
        raise RuntimeError(
            "Empty continuation tokenization."
        )

    return full_ids, target_ids


def score_candidate(
    engine: DecisionEngine,
    prefix: str,
    candidate: str,
) -> float:
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

    return score


def score_order(
    engine: DecisionEngine,
    state: str,
    question: str,
    reversed_order: bool,
) -> dict:
    prefix = make_prefix(
        engine.tokenizer,
        state,
        question,
        reversed_order,
    )

    score_true = score_candidate(
        engine,
        prefix,
        "True",
    )

    score_false = score_candidate(
        engine,
        prefix,
        "False",
    )

    delta = (
        score_true
        - score_false
    )

    return {
        "score_true": score_true,
        "score_false": score_false,
        "delta": delta,
        "probability_true": sigmoid(delta),
        "selected": delta >= 0.0,
    }


def metrics(
    deltas: list[float],
    targets: list[bool],
) -> dict:
    probabilities = [
        sigmoid(delta)
        for delta in deltas
    ]

    predictions = [
        delta >= 0.0
        for delta in deltas
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
            min(
                max(
                    probability
                    if target
                    else 1.0 - probability,
                    eps,
                ),
                1.0,
            )
        )
        for probability, target
        in zip(probabilities, targets)
    )

    return {
        "accuracy": accuracy,
        "brier": brier,
        "nll": nll,
        "predicted_true_rate": (
            sum(predictions)
            / len(predictions)
        ),
    }


def main() -> None:
    examples = json.loads(
        DATASET.read_text(
            encoding="utf-8"
        )
    )

    output_models = []

    for model_id in MODEL_IDS:
        print()
        print("=" * 80)
        print(model_id)
        print("=" * 80)

        engine = DecisionEngine(
            model=model_id
        )

        rows = []

        for index, example in enumerate(
            examples,
            start=1,
        ):
            normal = score_order(
                engine,
                example["state"],
                example["question"],
                reversed_order=False,
            )

            reversed_result = score_order(
                engine,
                example["state"],
                example["question"],
                reversed_order=True,
            )

            symmetric_delta = (
                normal["delta"]
                + reversed_result["delta"]
            ) / 2.0

            rows.append(
                {
                    "id": example["id"],
                    "category": example["category"],
                    "target": example["target"],
                    "normal": normal,
                    "reversed": reversed_result,
                    "symmetric_delta": (
                        symmetric_delta
                    ),
                    "symmetric_probability_true": (
                        sigmoid(
                            symmetric_delta
                        )
                    ),
                    "symmetric_selected": (
                        symmetric_delta >= 0.0
                    ),
                }
            )

            print(
                f"{index:3d}/120  "
                f'{example["id"]:<40} '
                f'target={str(example["target"]):5}  '
                f'normal={str(normal["selected"]):5}  '
                f'reversed={str(reversed_result["selected"]):5}  '
                f'sym={str(symmetric_delta >= 0.0):5}'
            )

        targets = [
            bool(row["target"])
            for row in rows
        ]

        normal_metrics = metrics(
            [
                row["normal"]["delta"]
                for row in rows
            ],
            targets,
        )

        reversed_metrics = metrics(
            [
                row["reversed"]["delta"]
                for row in rows
            ],
            targets,
        )

        symmetric_metrics = metrics(
            [
                row["symmetric_delta"]
                for row in rows
            ],
            targets,
        )

        order_changes = sum(
            row["normal"]["selected"]
            != row["reversed"]["selected"]
            for row in rows
        )

        mean_abs_delta_shift = (
            statistics.mean(
                abs(
                    row["normal"]["delta"]
                    - row["reversed"]["delta"]
                )
                for row in rows
            )
        )

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
                if row["category"] == category
            ]

            categories.append(
                {
                    "category": category,
                    "normal_accuracy": (
                        sum(
                            row["normal"]["selected"]
                            == row["target"]
                            for row in subset
                        )
                        / len(subset)
                    ),
                    "reversed_accuracy": (
                        sum(
                            row["reversed"]["selected"]
                            == row["target"]
                            for row in subset
                        )
                        / len(subset)
                    ),
                    "symmetric_accuracy": (
                        sum(
                            row["symmetric_selected"]
                            == row["target"]
                            for row in subset
                        )
                        / len(subset)
                    ),
                }
            )

        print()
        print("--- RESULTS ---")
        print(
            f'{"Method":<16}'
            f'{"Accuracy":>10}'
            f'{"Brier":>10}'
            f'{"NLL":>10}'
            f'{"True rate":>12}'
        )

        for name, result in (
            ("normal", normal_metrics),
            ("reversed", reversed_metrics),
            ("symmetric", symmetric_metrics),
        ):
            print(
                f'{name:<16}'
                f'{result["accuracy"]:>10.3f}'
                f'{result["brier"]:>10.3f}'
                f'{result["nll"]:>10.3f}'
                f'{result["predicted_true_rate"]:>12.3f}'
            )

        print()
        print(
            "Order changes:       ",
            f"{order_changes}/120",
        )
        print(
            "Mean |delta shift|:  ",
            f"{mean_abs_delta_shift:.6f}",
        )

        print()
        print("--- BY CATEGORY ---")
        print(
            f'{"Category":<16}'
            f'{"Normal":>10}'
            f'{"Reverse":>10}'
            f'{"Sym":>10}'
        )

        for row in categories:
            print(
                f'{row["category"]:<16}'
                f'{row["normal_accuracy"]:>10.3f}'
                f'{row["reversed_accuracy"]:>10.3f}'
                f'{row["symmetric_accuracy"]:>10.3f}'
            )

        output_models.append(
            {
                "model": model_id,
                "normal": normal_metrics,
                "reversed": reversed_metrics,
                "symmetric": symmetric_metrics,
                "order_changed_decisions": (
                    order_changes
                ),
                "mean_absolute_delta_shift": (
                    mean_abs_delta_shift
                ),
                "categories": categories,
                "examples": rows,
            }
        )

        del engine
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    OUTPUT.write_text(
        json.dumps(
            {
                "evaluation_type": (
                    "calibration_only_prompt_order_symmetry"
                ),
                "test_split_used": False,
                "generated_output_tokens": 0,
                "models": output_models,
            },
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
        f'{"Normal":>10}'
        f'{"Reverse":>10}'
        f'{"Symmetric":>12}'
        f'{"Changes":>10}'
    )

    for model in output_models:
        print(
            f'{model["model"]:<34}'
            f'{model["normal"]["accuracy"]:>10.3f}'
            f'{model["reversed"]["accuracy"]:>10.3f}'
            f'{model["symmetric"]["accuracy"]:>12.3f}'
            f'{model["order_changed_decisions"]:>10}'
        )

    print()
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()
