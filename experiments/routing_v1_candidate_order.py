from __future__ import annotations

import gc
import hashlib
import json
import statistics
from pathlib import Path

import torch

from llm_decision_engine import DecisionEngine


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "benchmarks" / "data" / "routing_v1_calibration.json"
BASELINE = ROOT / "results" / "routing_v1_definition_guided_calibration.json"
OUTPUT = ROOT / "results" / "routing_v1_candidate_order_calibration.json"

EXPECTED_SHA256 = "7f94ca16628c3591a69eb38342ee038635b0ef72f04f045262fc6145646759c0"

MODELS = (
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
)

QUESTION = "Which support category best matches this customer message?"


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_state(text, candidates, definitions):
    policy = "\n".join(
        f"{candidate}: {definitions[candidate]}"
        for candidate in candidates
    )
    return (
        "ROUTING POLICY:\n"
        + policy
        + "\n\nCUSTOMER MESSAGE:\n"
        + text
    )


def acc(rows):
    return statistics.mean(
        row["selected"] == row["target"]
        for row in rows
    )


def main():
    dataset_hash = sha256_file(DATASET)
    if dataset_hash != EXPECTED_SHA256:
        raise RuntimeError("Calibration dataset SHA256 mismatch.")

    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))

    if dataset["split"] != "calibration":
        raise RuntimeError("Expected calibration split.")

    if baseline["dataset_sha256"] != EXPECTED_SHA256:
        raise RuntimeError("Stored baseline SHA256 mismatch.")

    candidates = list(dataset["candidates"])
    definitions = dict(dataset["intent_definitions"])
    examples = dataset["examples"]

    orders = {
        "reversed": list(reversed(candidates)),
        "rotated_2": candidates[2:] + candidates[:2],
        "fixed_shuffle": [
            candidates[i]
            for i in (3, 5, 1, 4, 0, 2)
        ],
    }

    for name, order in orders.items():
        if len(order) != len(candidates) or set(order) != set(candidates):
            raise RuntimeError(f"Invalid order: {name}")

    baseline_models = {
        item["model"]: item
        for item in baseline["models"]
    }

    print("ROUTING V1 CANDIDATE-ORDER CALIBRATION")
    print("======================================")
    print()
    print("Dataset SHA256:", dataset_hash)
    print("Test split used: False")
    print("Policy-definition order: canonical and fixed")
    print()

    print("canonical:")
    print(" ", " | ".join(candidates))

    for name, order in orders.items():
        print(name + ":")
        print(" ", " | ".join(order))

    final_models = []

    for model_id in MODELS:
        print()
        print("=" * 80)
        print("MODEL:", model_id)
        print("=" * 80)

        baseline_examples = baseline_models[model_id]["definition_guided"]["examples"]

        baseline_by_id = {
            row["id"]: row
            for row in baseline_examples
        }

        canonical_accuracy = statistics.mean(
            baseline_by_id[e["id"]]["selected"] == e["intent"]
            for e in examples
        )

        engine = DecisionEngine(model=model_id)

        variant_results = {}
        selections = {
            e["id"]: [baseline_by_id[e["id"]]["selected"]]
            for e in examples
        }

        for variant_name, order in orders.items():
            print()
            print("---", variant_name.upper(), "---")

            rows = []
            abs_probability_deltas = []

            for index, example in enumerate(examples, start=1):
                state = build_state(
                    example["text"],
                    candidates,
                    definitions,
                )

                result = engine.choice(
                    state=state,
                    question=QUESTION,
                    candidates=order,
                    scoring="sum",
                    execution="sequential",
                )

                if result.generated_output_tokens != 0:
                    raise RuntimeError("Unexpected generated output tokens.")

                selected = str(result.selected)
                canonical_row = baseline_by_id[example["id"]]
                canonical_selected = str(canonical_row["selected"])

                probabilities = {
                    candidate: float(result.probabilities[candidate])
                    for candidate in candidates
                }

                for candidate in candidates:
                    abs_probability_deltas.append(
                        abs(
                            probabilities[candidate]
                            - float(canonical_row["probabilities"][candidate])
                        )
                    )

                same = selected == canonical_selected

                rows.append(
                    {
                        "id": example["id"],
                        "target": example["intent"],
                        "canonical_selected": canonical_selected,
                        "selected": selected,
                        "same_decision": same,
                        "correct": selected == example["intent"],
                        "probabilities": probabilities,
                    }
                )

                selections[example["id"]].append(selected)

                marker = "SAME" if same else "SWITCH"

                print(
                    f"{index:3d}/120  "
                    f"{example['id']:<52} "
                    f"canonical={canonical_selected:<18} "
                    f"variant={selected:<18} "
                    f"{marker}"
                )

            changed_rows = [
                row for row in rows
                if not row["same_decision"]
            ]

            fixed = 0
            broken = 0
            neutral = 0

            for row in changed_rows:
                target = row["target"]
                canonical_correct = row["canonical_selected"] == target
                variant_correct = row["selected"] == target

                if not canonical_correct and variant_correct:
                    fixed += 1
                elif canonical_correct and not variant_correct:
                    broken += 1
                else:
                    neutral += 1

            variant_accuracy = acc(rows)
            agreement = statistics.mean(
                row["same_decision"]
                for row in rows
            )

            mean_delta = statistics.mean(abs_probability_deltas)
            max_delta = max(abs_probability_deltas)

            print()
            print("Accuracy:                ", f"{variant_accuracy:.6f}")
            print("Canonical agreement:     ", f"{agreement:.6f}")
            print("Changed decisions:       ", len(changed_rows))
            print("Fixed errors:            ", fixed)
            print("Broken correct:          ", broken)
            print("Neutral switches:        ", neutral)
            print("Mean abs probability Δ:  ", f"{mean_delta:.6f}")
            print("Max abs probability Δ:   ", f"{max_delta:.6f}")

            if changed_rows:
                print()
                print("--- CHANGED DECISIONS ---")
                for row in changed_rows:
                    print(
                        row["id"],
                        "target=" + repr(row["target"]),
                        "canonical=" + repr(row["canonical_selected"]),
                        "variant=" + repr(row["selected"]),
                    )

            variant_results[variant_name] = {
                "candidate_order": order,
                "accuracy": variant_accuracy,
                "canonical_agreement": agreement,
                "changed_decisions": len(changed_rows),
                "fixed_errors": fixed,
                "broken_correct": broken,
                "neutral_switches": neutral,
                "mean_abs_probability_delta": mean_delta,
                "max_abs_probability_delta": max_delta,
                "examples": rows,
            }

        stable_count = sum(
            len(set(selections[e["id"]])) == 1
            for e in examples
        )

        stable_rate = stable_count / len(examples)

        print()
        print("--- MODEL ORDER STABILITY ---")
        print("Canonical accuracy:      ", f"{canonical_accuracy:.6f}")
        print(
            "Stable across ALL orders:",
            f"{stable_count}/{len(examples)}",
            f"({stable_rate:.6f})",
        )

        final_models.append(
            {
                "model": model_id,
                "canonical": {
                    "candidate_order": candidates,
                    "accuracy": canonical_accuracy,
                },
                "variants": variant_results,
                "stable_across_all_orders": {
                    "count": stable_count,
                    "rate": stable_rate,
                },
                "generated_output_tokens": 0,
            }
        )

        del engine
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    payload = {
        "benchmark": "routing_v1",
        "evaluation_type": "calibration_only_candidate_order_sensitivity",
        "split": "calibration",
        "test_split_used": False,
        "dataset_sha256": dataset_hash,
        "question": QUESTION,
        "scoring": "sum",
        "execution": "sequential",
        "policy_definition_order": "canonical_fixed",
        "canonical_candidate_order": candidates,
        "candidate_order_variants": orders,
        "generated_output_tokens": 0,
        "models": final_models,
    }

    OUTPUT.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("FINAL CANDIDATE-ORDER SENSITIVITY")
    print("=" * 80)
    print(
        f"{'Model':<34}"
        f"{'Canonical':>11}"
        f"{'Reverse':>10}"
        f"{'Rotate':>10}"
        f"{'Shuffle':>10}"
        f"{'All stable':>12}"
    )

    for item in final_models:
        variants = item["variants"]
        print(
            f"{item['model']:<34}"
            f"{item['canonical']['accuracy']:>11.3f}"
            f"{variants['reversed']['accuracy']:>10.3f}"
            f"{variants['rotated_2']['accuracy']:>10.3f}"
            f"{variants['fixed_shuffle']['accuracy']:>10.3f}"
            f"{item['stable_across_all_orders']['rate']:>12.3f}"
        )

    print()
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()
