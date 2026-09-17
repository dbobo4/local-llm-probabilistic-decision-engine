from __future__ import annotations

import gc
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path

import torch

from llm_decision_engine import DecisionEngine


ROOT = Path(__file__).resolve().parents[1]

DATASET = (
    ROOT
    / "benchmarks"
    / "data"
    / "routing_v1_test.json"
)

OUTPUT = (
    ROOT
    / "results"
    / "routing_v1_frozen_test.json"
)

EXPECTED_DATASET_SHA256 = (
    "abf06b95a126d122822a5cb17e0141d009e59d24de58e01878fcd834300cc6ee"
)

MODEL_IDS = (
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
)

EXPECTED_CANDIDATES = [
    "billing",
    "technical support",
    "sales",
    "account access",
    "cancellation",
    "shipping",
]

EXPECTED_DEFINITIONS = {
    "billing": (
        "Charges, invoices, payments, receipts, refunds, "
        "and unexpected fees."
    ),
    "technical support": (
        "Product failures, crashes, errors, broken features, "
        "synchronization, or other technical problems."
    ),
    "sales": (
        "Pre-purchase questions about pricing, quotes, demos, "
        "plans, licenses, or volume purchases."
    ),
    "account access": (
        "Login, password, authentication, locked-account, "
        "or two-factor-access problems."
    ),
    "cancellation": (
        "Requests to end, discontinue, or stop renewal of an "
        "existing subscription or service."
    ),
    "shipping": (
        "Delivery, parcels, tracking, courier status, "
        "delivery address, or arrival-time questions."
    ),
}

QUESTION = (
    "Which support category best matches this customer message?"
)

ECE_BINS = 10


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def build_guided_state(
    text: str,
) -> str:
    policy = "\n".join(
        f"{candidate}: {EXPECTED_DEFINITIONS[candidate]}"
        for candidate in EXPECTED_CANDIDATES
    )

    return (
        "ROUTING POLICY:\n"
        + policy
        + "\n\nCUSTOMER MESSAGE:\n"
        + text
    )


def build_choice_user_prompt(
    state: str,
) -> str:
    candidate_lines = "\n".join(
        f"- {candidate}"
        for candidate in EXPECTED_CANDIDATES
    )

    return (
        "STATE:\n"
        + state
        + "\n\nQUESTION:\n"
        + QUESTION
        + "\n\nCANDIDATES:\n"
        + candidate_lines
        + "\n\nReturn exactly one candidate."
    )


def parse_candidate(
    text: str,
) -> str | None:
    cleaned = text.strip().casefold()

    canonical = {
        candidate.casefold(): candidate
        for candidate in EXPECTED_CANDIDATES
    }

    return canonical.get(cleaned)


def top_label_ece(
    rows: list[dict],
    bins: int = ECE_BINS,
) -> float:
    buckets = [
        []
        for _ in range(bins)
    ]

    for row in rows:
        confidence = row["confidence"]

        index = min(
            int(confidence * bins),
            bins - 1,
        )

        buckets[index].append(row)

    total = len(rows)
    ece = 0.0

    for bucket in buckets:
        if not bucket:
            continue

        mean_confidence = statistics.mean(
            row["confidence"]
            for row in bucket
        )

        accuracy = statistics.mean(
            1.0 if row["correct"] else 0.0
            for row in bucket
        )

        ece += (
            len(bucket)
            / total
            * abs(
                accuracy
                - mean_confidence
            )
        )

    return ece


def evaluate_direct_rows(
    rows: list[dict],
) -> dict:
    n = len(rows)

    accuracy = statistics.mean(
        1.0 if row["correct"] else 0.0
        for row in rows
    )

    top2_accuracy = statistics.mean(
        1.0 if row["top2_correct"] else 0.0
        for row in rows
    )

    nll = statistics.mean(
        -math.log(
            max(
                row["probabilities"][
                    row["target"]
                ],
                1e-300,
            )
        )
        for row in rows
    )

    brier = statistics.mean(
        sum(
            (
                row["probabilities"][candidate]
                - (
                    1.0
                    if candidate == row["target"]
                    else 0.0
                )
            )
            ** 2
            for candidate in EXPECTED_CANDIDATES
        )
        for row in rows
    )

    mean_confidence = statistics.mean(
        row["confidence"]
        for row in rows
    )

    ece = top_label_ece(
        rows
    )

    per_class = {}

    for candidate in EXPECTED_CANDIDATES:
        class_rows = [
            row
            for row in rows
            if row["target"] == candidate
        ]

        per_class[candidate] = (
            sum(
                row["correct"]
                for row in class_rows
            )
            / len(class_rows)
        )

    predicted_counts = dict(
        Counter(
            row["selected"]
            for row in rows
        )
    )

    return {
        "examples": n,
        "accuracy": accuracy,
        "top2_accuracy": top2_accuracy,
        "multiclass_brier_sum": brier,
        "negative_log_likelihood": nll,
        "top_label_ece_10_bins": ece,
        "mean_confidence": mean_confidence,
        "per_class_accuracy": per_class,
        "predicted_counts": predicted_counts,
    }


def main() -> None:
    dataset_hash = sha256_file(
        DATASET
    )

    if dataset_hash != EXPECTED_DATASET_SHA256:
        raise RuntimeError(
            "Frozen test dataset SHA256 mismatch."
        )

    dataset = json.loads(
        DATASET.read_text(
            encoding="utf-8"
        )
    )

    if dataset["split"] != "test":
        raise RuntimeError(
            "Expected routing_v1 test split."
        )

    if list(dataset["candidates"]) != EXPECTED_CANDIDATES:
        raise RuntimeError(
            "Frozen candidate order mismatch."
        )

    if dict(dataset["intent_definitions"]) != EXPECTED_DEFINITIONS:
        raise RuntimeError(
            "Frozen intent definitions mismatch."
        )

    examples = dataset["examples"]

    if len(examples) != 120:
        raise RuntimeError(
            "Expected exactly 120 frozen test examples."
        )

    class_counts = Counter(
        example["intent"]
        for example in examples
    )

    expected_class_counts = {
        candidate: 20
        for candidate in EXPECTED_CANDIDATES
    }

    if dict(class_counts) != expected_class_counts:
        raise RuntimeError(
            "Frozen test class distribution mismatch."
        )

    print("ROUTING V1 FROZEN TEST")
    print("======================")
    print()
    print("Dataset SHA256:", dataset_hash)
    print("Examples:", len(examples))
    print("Split: test")
    print("Protocol frozen before model evaluation: True")
    print("Scoring: sum")
    print("Execution: sequential")
    print("Candidate order: canonical")
    print("Direct generation: 0 output tokens")
    print()

    output_models = []

    for model_id in MODEL_IDS:
        print("=" * 80)
        print("MODEL:", model_id)
        print("=" * 80)

        engine = DecisionEngine(
            model=model_id
        )

        model = engine.model
        tokenizer = engine.tokenizer

        model.eval()

        input_device = (
            model
            .get_input_embeddings()
            .weight
            .device
        )

        direct_rows = []
        greedy_rows = []

        direct_generated_tokens = 0
        greedy_generated_tokens = 0

        for index, example in enumerate(
            examples,
            start=1,
        ):
            state = build_guided_state(
                example["text"]
            )

            direct = engine.choice(
                state=state,
                question=QUESTION,
                candidates=EXPECTED_CANDIDATES,
                scoring="sum",
                execution="sequential",
            )

            if direct.generated_output_tokens != 0:
                raise RuntimeError(
                    "Direct method unexpectedly generated output tokens."
                )

            direct_generated_tokens += (
                direct.generated_output_tokens
            )

            probabilities = {
                candidate: float(
                    direct.probabilities[
                        candidate
                    ]
                )
                for candidate in EXPECTED_CANDIDATES
            }

            selected = str(
                direct.selected
            )

            ranked = sorted(
                EXPECTED_CANDIDATES,
                key=lambda candidate: probabilities[
                    candidate
                ],
                reverse=True,
            )

            confidence = probabilities[
                selected
            ]

            direct_rows.append(
                {
                    "id": example["id"],
                    "target": example["intent"],
                    "selected": selected,
                    "correct": (
                        selected
                        == example["intent"]
                    ),
                    "top2": ranked[:2],
                    "top2_correct": (
                        example["intent"]
                        in ranked[:2]
                    ),
                    "confidence": confidence,
                    "scores": {
                        candidate: float(
                            direct.scores[
                                candidate
                            ]
                        )
                        for candidate
                        in EXPECTED_CANDIDATES
                    },
                    "probabilities": probabilities,
                    "generated_output_tokens": 0,
                }
            )

            user_prompt = (
                build_choice_user_prompt(
                    state
                )
            )

            prompt = (
                tokenizer.apply_chat_template(
                    [
                        {
                            "role": "user",
                            "content": user_prompt,
                        }
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

            encoded = tokenizer(
                prompt,
                return_tensors="pt",
            )

            encoded = {
                key: value.to(
                    input_device
                )
                for key, value
                in encoded.items()
            }

            prompt_length = (
                encoded["input_ids"].shape[1]
            )

            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=8,
                    pad_token_id=(
                        tokenizer.eos_token_id
                    ),
                )

            continuation_ids = (
                generated[
                    0,
                    prompt_length:
                ]
            )

            generated_token_count = int(
                continuation_ids.shape[0]
            )

            greedy_generated_tokens += (
                generated_token_count
            )

            raw_text = tokenizer.decode(
                continuation_ids,
                skip_special_tokens=True,
            )

            greedy_selected = (
                parse_candidate(
                    raw_text
                )
            )

            greedy_rows.append(
                {
                    "id": example["id"],
                    "target": example["intent"],
                    "selected": greedy_selected,
                    "raw_text": raw_text,
                    "parsed": (
                        greedy_selected
                        is not None
                    ),
                    "correct": (
                        greedy_selected
                        == example["intent"]
                    ),
                    "generated_output_tokens": (
                        generated_token_count
                    ),
                }
            )

            agreement = (
                selected
                == greedy_selected
            )

            print(
                f"{index:3d}/120  "
                f"{example['id']:<52} "
                f"target={example['intent']:<18} "
                f"direct={selected:<18} "
                f"greedy={str(greedy_selected):<18} "
                f"{'AGREE' if agreement else 'DIFF'}"
            )

        direct_metrics = (
            evaluate_direct_rows(
                direct_rows
            )
        )

        greedy_parse_rate = (
            sum(
                row["parsed"]
                for row in greedy_rows
            )
            / len(greedy_rows)
        )

        greedy_accuracy = (
            sum(
                row["correct"]
                for row in greedy_rows
            )
            / len(greedy_rows)
        )

        parsed_rows = [
            row
            for row in greedy_rows
            if row["parsed"]
        ]

        greedy_accuracy_parsed = (
            sum(
                row["correct"]
                for row in parsed_rows
            )
            / len(parsed_rows)
            if parsed_rows
            else None
        )

        agreement_rate = (
            sum(
                direct_row["selected"]
                == greedy_row["selected"]
                for direct_row, greedy_row
                in zip(
                    direct_rows,
                    greedy_rows,
                    strict=True,
                )
            )
            / len(direct_rows)
        )

        differing = []

        for direct_row, greedy_row in zip(
            direct_rows,
            greedy_rows,
            strict=True,
        ):
            if (
                direct_row["selected"]
                != greedy_row["selected"]
            ):
                differing.append(
                    {
                        "id": direct_row["id"],
                        "target": direct_row[
                            "target"
                        ],
                        "direct": direct_row[
                            "selected"
                        ],
                        "greedy": greedy_row[
                            "selected"
                        ],
                        "greedy_raw_text": (
                            greedy_row[
                                "raw_text"
                            ]
                        ),
                    }
                )

        print()
        print("--- DIRECT ---")
        print(
            "Accuracy:                 ",
            f"{direct_metrics['accuracy']:.6f}",
        )
        print(
            "Top-2 accuracy:           ",
            f"{direct_metrics['top2_accuracy']:.6f}",
        )
        print(
            "Brier sum:                ",
            f"{direct_metrics['multiclass_brier_sum']:.6f}",
        )
        print(
            "NLL:                      ",
            f"{direct_metrics['negative_log_likelihood']:.6f}",
        )
        print(
            "ECE (10 bins):            ",
            f"{direct_metrics['top_label_ece_10_bins']:.6f}",
        )
        print(
            "Mean confidence:          ",
            f"{direct_metrics['mean_confidence']:.6f}",
        )
        print(
            "Generated output tokens:  ",
            direct_generated_tokens,
        )

        print()
        print("--- GREEDY CONTROL ---")
        print(
            "Accuracy:                 ",
            f"{greedy_accuracy:.6f}",
        )
        print(
            "Exact parse rate:         ",
            f"{greedy_parse_rate:.6f}",
        )
        print(
            "Accuracy parsed only:     ",
            (
                f"{greedy_accuracy_parsed:.6f}"
                if greedy_accuracy_parsed
                is not None
                else "N/A"
            ),
        )
        print(
            "Direct/greedy agreement:  ",
            f"{agreement_rate:.6f}",
        )
        print(
            "Different decisions:      ",
            len(differing),
        )
        print(
            "Generated output tokens:  ",
            greedy_generated_tokens,
        )

        output_models.append(
            {
                "model": model_id,
                "direct": {
                    **direct_metrics,
                    "generated_output_tokens": (
                        direct_generated_tokens
                    ),
                    "examples_detail": (
                        direct_rows
                    ),
                },
                "greedy_control": {
                    "accuracy": (
                        greedy_accuracy
                    ),
                    "exact_parse_rate": (
                        greedy_parse_rate
                    ),
                    "accuracy_parsed_only": (
                        greedy_accuracy_parsed
                    ),
                    "direct_agreement": (
                        agreement_rate
                    ),
                    "different_decisions": (
                        len(differing)
                    ),
                    "generated_output_tokens": (
                        greedy_generated_tokens
                    ),
                    "differences": differing,
                    "examples_detail": (
                        greedy_rows
                    ),
                },
            }
        )

        del engine
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print()

    payload = {
        "benchmark": "routing_v1",
        "evaluation_type": (
            "frozen_pre_specified_held_out_test"
        ),
        "split": "test",
        "dataset_sha256": (
            dataset_hash
        ),
        "protocol": {
            "definition_guided": True,
            "question": QUESTION,
            "scoring": "sum",
            "execution": "sequential",
            "candidate_order": (
                EXPECTED_CANDIDATES
            ),
            "intent_definitions": (
                EXPECTED_DEFINITIONS
            ),
            "direct_generated_output_tokens": 0,
            "greedy_control": {
                "do_sample": False,
                "max_new_tokens": 8,
                "parser": (
                    "strip + casefold + exact candidate match"
                ),
            },
        },
        "models": output_models,
    }

    OUTPUT.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print("=" * 80)
    print("FINAL FROZEN TEST SUMMARY")
    print("=" * 80)

    print(
        f"{'Model':<34}"
        f"{'Direct':>10}"
        f"{'Top2':>10}"
        f"{'Greedy':>10}"
        f"{'Agree':>10}"
        f"{'D tokens':>11}"
        f"{'G tokens':>11}"
    )

    for item in output_models:
        print(
            f"{item['model']:<34}"
            f"{item['direct']['accuracy']:>10.3f}"
            f"{item['direct']['top2_accuracy']:>10.3f}"
            f"{item['greedy_control']['accuracy']:>10.3f}"
            f"{item['greedy_control']['direct_agreement']:>10.3f}"
            f"{item['direct']['generated_output_tokens']:>11}"
            f"{item['greedy_control']['generated_output_tokens']:>11}"
        )

    print()
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()
