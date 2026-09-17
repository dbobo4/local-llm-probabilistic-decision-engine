from __future__ import annotations

import gc
import hashlib
import json
import statistics
import time
from collections import Counter
from pathlib import Path

import torch

from llm_decision_engine import DecisionEngine


ROOT = Path(__file__).resolve().parents[1]

DATASET = (
    ROOT
    / "benchmarks"
    / "data"
    / "routing_v1_calibration.json"
)

DIRECT_RESULTS = (
    ROOT
    / "results"
    / "routing_v1_definition_guided_calibration.json"
)

OUTPUT = (
    ROOT
    / "results"
    / "routing_v1_direct_vs_greedy_calibration.json"
)

EXPECTED_DATASET_SHA256 = (
    "7f94ca16628c3591a69eb38342ee0386"
    "35b0ef72f04f045262fc6145646759c0"
)

MODEL_IDS = (
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
)

ROUTING_QUESTION = (
    "Which support category best matches this customer message?"
)


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
    candidates: list[str],
    definitions: dict[str, str],
) -> str:
    policy = "\n".join(
        f"{candidate}: {definitions[candidate]}"
        for candidate in candidates
    )

    return (
        "ROUTING POLICY:\n"
        f"{policy}\n\n"
        "CUSTOMER MESSAGE:\n"
        f"{text}"
    )


def build_choice_user_prompt(
    state: str,
    question: str,
    candidates: list[str],
) -> str:
    candidate_list = "\n".join(
        f"- {candidate}"
        for candidate in candidates
    )

    return f"""STATE:
{state}

QUESTION:
{question}

CANDIDATES:
{candidate_list}

Return exactly one candidate."""


def parse_exact_candidate(
    text: str,
    candidates: list[str],
) -> str | None:
    cleaned = text.strip()

    canonical = {
        candidate.casefold(): candidate
        for candidate in candidates
    }

    return canonical.get(
        cleaned.casefold()
    )


def main() -> None:
    dataset_hash = sha256_file(
        DATASET
    )

    if dataset_hash != EXPECTED_DATASET_SHA256:
        raise RuntimeError(
            "Calibration dataset SHA256 mismatch."
        )

    dataset_payload = json.loads(
        DATASET.read_text(
            encoding="utf-8"
        )
    )

    if dataset_payload["split"] != "calibration":
        raise RuntimeError(
            "Expected calibration split."
        )

    candidates = list(
        dataset_payload["candidates"]
    )

    definitions = dict(
        dataset_payload["intent_definitions"]
    )

    examples = dataset_payload[
        "examples"
    ]

    direct_payload = json.loads(
        DIRECT_RESULTS.read_text(
            encoding="utf-8"
        )
    )

    if (
        direct_payload["dataset_sha256"]
        != EXPECTED_DATASET_SHA256
    ):
        raise RuntimeError(
            "Stored direct result dataset SHA mismatch."
        )

    direct_models = {
        model["model"]: model
        for model in direct_payload["models"]
    }

    print("ROUTING V1 DIRECT VS GREEDY CALIBRATION")
    print("=======================================")
    print()
    print("Dataset SHA256:", dataset_hash)
    print("Examples:", len(examples))
    print("Test split used: False")
    print("Scoring: direct=sum")
    print("Generation: greedy, do_sample=False")
    print()

    output_models = []

    for model_id in MODEL_IDS:
        print("=" * 80)
        print("MODEL:", model_id)
        print("=" * 80)

        direct_model = direct_models[
            model_id
        ]

        direct_rows_by_id = {
            row["id"]: row
            for row in direct_model[
                "definition_guided"
            ]["examples"]
        }

        engine = DecisionEngine(
            model=model_id
        )

        model = engine.model
        tokenizer = engine.tokenizer

        greedy_rows = []

        total_generated_tokens = 0

        for index, example in enumerate(
            examples,
            start=1,
        ):
            state = build_guided_state(
                example["text"],
                candidates,
                definitions,
            )

            user_prompt = build_choice_user_prompt(
                state,
                ROUTING_QUESTION,
                candidates,
            )

            prefix = tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": user_prompt,
                    }
                ],
                tokenize=False,
                add_generation_prompt=True,
            )

            encoded = tokenizer(
                prefix,
                return_tensors="pt",
                add_special_tokens=False,
            )

            input_device = next(
                model.parameters()
            ).device

            encoded = {
                key: value.to(input_device)
                for key, value in encoded.items()
            }

            input_length = int(
                encoded["input_ids"].shape[1]
            )

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            started = time.perf_counter()

            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=8,
                    pad_token_id=tokenizer.eos_token_id,
                )

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            latency_ms = (
                time.perf_counter()
                - started
            ) * 1000.0

            continuation_ids = generated[
                0,
                input_length:,
            ]

            generated_token_count = int(
                continuation_ids.shape[0]
            )

            total_generated_tokens += (
                generated_token_count
            )

            decoded = tokenizer.decode(
                continuation_ids,
                skip_special_tokens=True,
            )

            parsed = parse_exact_candidate(
                decoded,
                candidates,
            )

            direct_selected = str(
                direct_rows_by_id[
                    example["id"]
                ]["selected"]
            )

            greedy_correct = (
                parsed == example["intent"]
            )

            agreement = (
                parsed == direct_selected
            )

            greedy_rows.append(
                {
                    "id": example["id"],
                    "family": example["family"],
                    "text": example["text"],
                    "target": example["intent"],
                    "direct_selected": direct_selected,
                    "raw_generated_text": decoded,
                    "greedy_selected": parsed,
                    "parsed": parsed is not None,
                    "greedy_correct": greedy_correct,
                    "direct_greedy_agreement": agreement,
                    "generated_output_tokens": (
                        generated_token_count
                    ),
                    "latency_ms": latency_ms,
                }
            )

            print(
                f"{index:3d}/120  "
                f'{example["id"]:<52} '
                f'target={example["intent"]:<18} '
                f'direct={direct_selected:<18} '
                f'greedy={str(parsed):<18} '
                f'tokens={generated_token_count:<2} '
                f'{"AGREE" if agreement else "DIFF"}'
            )

        parsed_rows = [
            row
            for row in greedy_rows
            if row["parsed"]
        ]

        parse_rate = (
            len(parsed_rows)
            / len(greedy_rows)
        )

        greedy_accuracy_all = statistics.mean(
            1.0
            if row["greedy_correct"]
            else 0.0
            for row in greedy_rows
        )

        greedy_accuracy_parsed = (
            statistics.mean(
                1.0
                if row["greedy_correct"]
                else 0.0
                for row in parsed_rows
            )
            if parsed_rows
            else float("nan")
        )

        direct_accuracy = statistics.mean(
            1.0
            if direct_rows_by_id[
                example["id"]
            ]["selected"]
            == example["intent"]
            else 0.0
            for example in examples
        )

        agreement_rate = statistics.mean(
            1.0
            if row["direct_greedy_agreement"]
            else 0.0
            for row in greedy_rows
        )

        greedy_per_class = {}

        for candidate in candidates:
            subset = [
                row
                for row in greedy_rows
                if row["target"] == candidate
            ]

            greedy_per_class[candidate] = (
                statistics.mean(
                    1.0
                    if row["greedy_selected"]
                    == candidate
                    else 0.0
                    for row in subset
                )
            )

        generated_counts = Counter(
            row["generated_output_tokens"]
            for row in greedy_rows
        )

        latencies = [
            row["latency_ms"]
            for row in greedy_rows
        ]

        differing = [
            row
            for row in greedy_rows
            if not row[
                "direct_greedy_agreement"
            ]
        ]

        print()
        print("--- SUMMARY ---")
        print(
            "Direct accuracy:             ",
            f"{direct_accuracy:.6f}",
        )
        print(
            "Greedy accuracy:             ",
            f"{greedy_accuracy_all:.6f}",
        )
        print(
            "Greedy exact parse rate:     ",
            f"{parse_rate:.6f}",
        )
        print(
            "Greedy acc among parsed:     ",
            f"{greedy_accuracy_parsed:.6f}",
        )
        print(
            "Direct/greedy agreement:     ",
            f"{agreement_rate:.6f}",
        )
        print(
            "Different decisions:         ",
            len(differing),
        )
        print(
            "Direct generated tokens:     ",
            0,
        )
        print(
            "Greedy generated tokens:     ",
            total_generated_tokens,
        )
        print(
            "Greedy mean output tokens:   ",
            f"{total_generated_tokens / len(greedy_rows):.3f}",
        )
        print(
            "Greedy median latency:       ",
            f"{statistics.median(latencies):.2f} ms",
        )

        print()
        print("--- GREEDY PER-CLASS ACCURACY ---")

        for candidate in candidates:
            print(
                f"{candidate:<20}"
                f"{greedy_per_class[candidate]:.3f}"
            )

        print()
        print("--- GENERATED TOKEN COUNTS ---")

        for token_count in sorted(
            generated_counts
        ):
            print(
                f"{token_count:>2} tokens: "
                f"{generated_counts[token_count]}"
            )

        if differing:
            print()
            print("--- DIRECT/GREEDY DIFFERENCES ---")

            for row in differing:
                print(
                    f'{row["id"]}: '
                    f'target={row["target"]!r}, '
                    f'direct={row["direct_selected"]!r}, '
                    f'greedy={row["greedy_selected"]!r}, '
                    f'raw={row["raw_generated_text"]!r}'
                )

        output_models.append(
            {
                "model": model_id,
                "direct": {
                    "accuracy": direct_accuracy,
                    "generated_output_tokens": 0,
                },
                "greedy": {
                    "accuracy_all_examples": (
                        greedy_accuracy_all
                    ),
                    "accuracy_parsed_only": (
                        greedy_accuracy_parsed
                    ),
                    "exact_parse_rate": parse_rate,
                    "generated_output_tokens": (
                        total_generated_tokens
                    ),
                    "mean_generated_output_tokens": (
                        total_generated_tokens
                        / len(greedy_rows)
                    ),
                    "median_latency_ms": (
                        statistics.median(
                            latencies
                        )
                    ),
                    "per_class_accuracy": (
                        greedy_per_class
                    ),
                },
                "direct_greedy_agreement": {
                    "rate": agreement_rate,
                    "different_decisions": (
                        len(differing)
                    ),
                },
                "examples": greedy_rows,
            }
        )

        del engine
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print()

    OUTPUT.write_text(
        json.dumps(
            {
                "benchmark": "routing_v1",
                "evaluation_type": (
                    "calibration_only_definition_guided_direct_vs_greedy"
                ),
                "split": "calibration",
                "test_split_used": False,
                "dataset_sha256": dataset_hash,
                "question": ROUTING_QUESTION,
                "candidates": candidates,
                "intent_definitions": definitions,
                "direct_scoring": "sum",
                "direct_execution": "sequential",
                "greedy_do_sample": False,
                "greedy_max_new_tokens": 8,
                "models": output_models,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print("=" * 80)
    print("FINAL DIRECT VS GREEDY")
    print("=" * 80)

    print(
        f'{"Model":<34}'
        f'{"Direct":>10}'
        f'{"Greedy":>10}'
        f'{"Parse":>10}'
        f'{"Agree":>10}'
        f'{"D tokens":>10}'
        f'{"G tokens":>10}'
    )

    for result in output_models:
        print(
            f'{result["model"]:<34}'
            f'{result["direct"]["accuracy"]:>10.3f}'
            f'{result["greedy"]["accuracy_all_examples"]:>10.3f}'
            f'{result["greedy"]["exact_parse_rate"]:>10.3f}'
            f'{result["direct_greedy_agreement"]["rate"]:>10.3f}'
            f'{result["direct"]["generated_output_tokens"]:>10}'
            f'{result["greedy"]["generated_output_tokens"]:>10}'
        )

    print()
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()
