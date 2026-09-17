from __future__ import annotations

import hashlib
import json
import statistics
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import torch

from llm_decision_engine import (
    DecisionEngine,
    evaluate_binary_calibration,
    evaluate_binary_predictions,
)


ROOT = Path(__file__).resolve().parents[1]

DATASET = (
    ROOT
    / "benchmarks"
    / "data"
    / "binary_quality_v2_test.json"
)

OUTPUT = (
    ROOT
    / "results"
    / "binary_quality_v2_test_final.json"
)

EXPECTED_DATASET_SHA256 = (
    "50ffd1d786995835c22e8e35d84545b7"
    "d76dec167ed3e259b87876ac8885e749"
)

PREREGISTERED_COMMIT = (
    "39bbd0b98dd4e89d72f4fe9a545a6633b5e085b0"
)

MODEL_IDS = (
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
)

REASONING_MAX_NEW_TOKENS = 96


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def make_structured_reasoning_prefix(
    tokenizer,
    state: str,
    question: str,
) -> str:
    content = (
        f"STATE:\n{state}\n\n"
        f"QUESTION:\n{question}\n\n"
        "Verify the statement without giving a verdict. "
        "Use exactly these three lines:\n"
        "GIVEN: restate the relevant mathematical claim.\n"
        "COMPUTED: independently compute or check the relevant value.\n"
        "REFERENCE: state the value, parity, divisibility condition, "
        "or relation claimed by the original statement.\n"
        "Do not use the words True, False, correct, incorrect, "
        "yes, or no."
    )

    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )


def generate_reasoning(
    engine: DecisionEngine,
    prefix: str,
) -> tuple[str, int, float]:
    encoded = engine.tokenizer(
        prefix,
        return_tensors="pt",
        add_special_tokens=False,
    )

    device = next(engine.model.parameters()).device

    encoded = {
        key: value.to(device)
        for key, value in encoded.items()
    }

    input_length = encoded["input_ids"].shape[1]

    synchronize()
    start = time.perf_counter()

    with torch.inference_mode():
        output = engine.model.generate(
            **encoded,
            max_new_tokens=REASONING_MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=engine.tokenizer.eos_token_id,
        )

    synchronize()
    elapsed = time.perf_counter() - start

    generated_ids = output[0, input_length:]

    text = engine.tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()

    return (
        text,
        int(generated_ids.numel()),
        elapsed,
    )


def binary_metrics(
    probabilities: list[float],
    targets: list[bool],
) -> dict:
    evaluation = evaluate_binary_predictions(
        probabilities,
        targets,
    )

    calibration = evaluate_binary_calibration(
        probabilities,
        targets,
        num_bins=10,
    )

    return {
        "evaluation": asdict(evaluation),
        "calibration": asdict(calibration),
    }


def run_model(
    model_id: str,
    examples: list[dict],
) -> dict:
    print()
    print("=" * 80)
    print("MODEL:", model_id)
    print("=" * 80)

    engine = DecisionEngine(model=model_id)

    targets = []

    direct_probabilities = []
    direct_choices = []

    reasoned_probabilities = []
    reasoned_choices = []

    reasoning_tokens = []
    reasoning_times = []

    rows = []

    for index, example in enumerate(examples, start=1):
        target = example["target"]

        direct = engine.boolean(
            state=example["state"],
            question=example["question"],
            scoring="sum",
            execution="sequential",
        )

        reasoning_prefix = make_structured_reasoning_prefix(
            engine.tokenizer,
            example["state"],
            example["question"],
        )

        rationale, tokens, seconds = generate_reasoning(
            engine,
            reasoning_prefix,
        )

        reasoning_state = (
            f"{example['state']}\n\n"
            "VERIFICATION WORK:\n"
            f"{rationale}"
        )

        reasoned = engine.boolean(
            state=reasoning_state,
            question=example["question"],
            scoring="sum",
            execution="sequential",
        )

        targets.append(target)

        direct_probabilities.append(
            direct.probability_true
        )

        direct_choices.append(
            direct.selected
        )

        reasoned_probabilities.append(
            reasoned.probability_true
        )

        reasoned_choices.append(
            reasoned.selected
        )

        reasoning_tokens.append(tokens)
        reasoning_times.append(seconds)

        rows.append(
            {
                "id": example["id"],
                "category": example["category"],
                "state": example["state"],
                "question": example["question"],
                "target": target,
                "direct": {
                    "probability_true": (
                        direct.probability_true
                    ),
                    "probability_false": (
                        direct.probability_false
                    ),
                    "score_true": (
                        direct.scores["yes"]
                    ),
                    "score_false": (
                        direct.scores["no"]
                    ),
                    "score_delta": (
                        direct.scores["yes"]
                        - direct.scores["no"]
                    ),
                    "selected": direct.selected,
                    "correct": (
                        direct.selected == target
                    ),
                    "generated_output_tokens": 0,
                },
                "structured_reasoning": {
                    "text": rationale,
                    "generated_output_tokens": tokens,
                    "seconds": seconds,
                    "hit_token_limit": (
                        tokens
                        >= REASONING_MAX_NEW_TOKENS
                    ),
                },
                "reasoning_conditioned_direct": {
                    "probability_true": (
                        reasoned.probability_true
                    ),
                    "probability_false": (
                        reasoned.probability_false
                    ),
                    "score_true": (
                        reasoned.scores["yes"]
                    ),
                    "score_false": (
                        reasoned.scores["no"]
                    ),
                    "score_delta": (
                        reasoned.scores["yes"]
                        - reasoned.scores["no"]
                    ),
                    "selected": reasoned.selected,
                    "correct": (
                        reasoned.selected == target
                    ),
                    "generated_output_tokens": 0,
                },
            }
        )

        print(
            f"{index:3d}/{len(examples)}  "
            f'{example["id"]:<34} '
            f"target={str(target):5}  "
            f"direct={str(direct.selected):5}  "
            f"reasoned={str(reasoned.selected):5}"
        )

    direct_metrics = binary_metrics(
        direct_probabilities,
        targets,
    )

    reasoned_metrics = binary_metrics(
        reasoned_probabilities,
        targets,
    )

    direct_accuracy = (
        direct_metrics["evaluation"]["accuracy"]
    )

    reasoned_accuracy = (
        reasoned_metrics["evaluation"]["accuracy"]
    )

    fixes = sum(
        direct != target
        and reasoned == target
        for direct, reasoned, target in zip(
            direct_choices,
            reasoned_choices,
            targets,
        )
    )

    breaks = sum(
        direct == target
        and reasoned != target
        for direct, reasoned, target in zip(
            direct_choices,
            reasoned_choices,
            targets,
        )
    )

    categories = []

    for category in sorted(
        {example["category"] for example in examples}
    ):
        subset = [
            row
            for row in rows
            if row["category"] == category
        ]

        direct_category = sum(
            row["direct"]["correct"]
            for row in subset
        ) / len(subset)

        reasoned_category = sum(
            row["reasoning_conditioned_direct"]["correct"]
            for row in subset
        ) / len(subset)

        categories.append(
            {
                "category": category,
                "count": len(subset),
                "direct_accuracy": direct_category,
                "reasoned_accuracy": reasoned_category,
            }
        )

    result = {
        "model": model_id,
        "direct": direct_metrics,
        "reasoning_conditioned": reasoned_metrics,
        "reasoning_fixes_direct_errors": fixes,
        "reasoning_breaks_direct_correct": breaks,
        "reasoning_generated_tokens_total": sum(
            reasoning_tokens
        ),
        "reasoning_generated_tokens_mean": (
            statistics.mean(reasoning_tokens)
        ),
        "reasoning_generated_tokens_median": (
            statistics.median(reasoning_tokens)
        ),
        "reasoning_latency_median_ms": (
            statistics.median(reasoning_times)
            * 1000.0
        ),
        "reasoning_hit_token_limit": sum(
            value >= REASONING_MAX_NEW_TOKENS
            for value in reasoning_tokens
        ),
        "categories": categories,
        "examples": rows,
    }

    print()
    print("--- HELD-OUT TEST SUMMARY ---")
    print(
        f"Direct accuracy:       "
        f"{direct_accuracy:.6f}"
    )
    print(
        f"Direct Brier:          "
        f"{direct_metrics['evaluation']['mean_brier']:.6f}"
    )
    print(
        f"Direct NLL:            "
        f"{direct_metrics['evaluation']['mean_nll']:.6f}"
    )
    print(
        f"Direct ECE:            "
        f"{direct_metrics['calibration']['expected_calibration_error']:.6f}"
    )

    print()
    print(
        f"Reasoned accuracy:     "
        f"{reasoned_accuracy:.6f}"
    )
    print(
        f"Reasoned Brier:        "
        f"{reasoned_metrics['evaluation']['mean_brier']:.6f}"
    )
    print(
        f"Reasoned NLL:          "
        f"{reasoned_metrics['evaluation']['mean_nll']:.6f}"
    )
    print(
        f"Reasoned ECE:          "
        f"{reasoned_metrics['calibration']['expected_calibration_error']:.6f}"
    )

    print()
    print(
        "Reasoning fixes:       ",
        fixes,
    )
    print(
        "Reasoning breaks:      ",
        breaks,
    )
    print(
        "Reasoning token mean:  ",
        f"{statistics.mean(reasoning_tokens):.2f}",
    )
    print(
        "Reasoning latency med: ",
        f"{statistics.median(reasoning_times) * 1000.0:.2f} ms",
    )

    print()
    print("--- BY CATEGORY ---")

    for row in categories:
        print(
            f'{row["category"]:<16}'
            f'direct={row["direct_accuracy"]:.3f}  '
            f'reasoned={row["reasoned_accuracy"]:.3f}'
        )

    del engine

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def main() -> None:
    dataset_hash = sha256_file(DATASET)

    if dataset_hash != EXPECTED_DATASET_SHA256:
        raise RuntimeError(
            "Frozen test split hash mismatch. "
            f"Expected {EXPECTED_DATASET_SHA256}, "
            f"got {dataset_hash}."
        )

    head = git_head()

    if head != PREREGISTERED_COMMIT:
        raise RuntimeError(
            "Repository HEAD differs from the preregistered "
            "pre-test commit. "
            f"Expected {PREREGISTERED_COMMIT}, got {head}."
        )

    examples = json.loads(
        DATASET.read_text(encoding="utf-8")
    )

    if len(examples) != 120:
        raise RuntimeError(
            f"Expected 120 held-out test examples, "
            f"got {len(examples)}."
        )

    print("HELD-OUT TEST EVALUATION")
    print("========================")
    print("Dataset:", DATASET.relative_to(ROOT))
    print("SHA256: ", dataset_hash)
    print("Commit: ", head)
    print("Examples:", len(examples))
    print()
    print(
        "Protocol was fixed before examining "
        "test results."
    )

    results = []

    for model_id in MODEL_IDS:
        results.append(
            run_model(
                model_id,
                examples,
            )
        )

    payload = {
        "evaluation_type": "held_out_test",
        "protocol_frozen_before_test": True,
        "preregistered_commit": head,
        "dataset": str(
            DATASET.relative_to(ROOT)
        ),
        "dataset_sha256": dataset_hash,
        "example_count": len(examples),
        "main_method": (
            "direct dedicated True/False probabilistic scoring"
        ),
        "main_method_generated_output_tokens": 0,
        "diagnostic_control": (
            "structured verification generation "
            "followed by direct scoring"
        ),
        "diagnostic_control_is_main_method": False,
        "reasoning_max_new_tokens": (
            REASONING_MAX_NEW_TOKENS
        ),
        "models": results,
    }

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("FINAL HELD-OUT COMPARISON")
    print("=" * 80)

    print(
        f'{"Model":<34}'
        f'{"Direct":>10}'
        f'{"Reasoned":>12}'
        f'{"Gain":>10}'
    )

    for result in results:
        direct_accuracy = (
            result["direct"]["evaluation"]["accuracy"]
        )

        reasoned_accuracy = (
            result[
                "reasoning_conditioned"
            ]["evaluation"]["accuracy"]
        )

        print(
            f'{result["model"]:<34}'
            f'{direct_accuracy:>10.3f}'
            f'{reasoned_accuracy:>12.3f}'
            f'{reasoned_accuracy - direct_accuracy:>+10.3f}'
        )

    print()
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()
