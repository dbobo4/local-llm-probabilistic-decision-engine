import json
import statistics
import time
from pathlib import Path

import torch

from llm_decision_engine import DecisionEngine
from llm_decision_engine.tokenization import tokenize_continuation


MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
CANDIDATE_COUNTS = [2, 3, 5, 10, 20]
ROUNDS = 30
WARMUP_ROUNDS = 2

state = "A request must be routed to exactly one available category."
question = "Which candidate should handle this request?"

engine = DecisionEngine(model=MODEL_ID)
device = next(engine.model.parameters()).device
tokenizer = engine.tokenizer


def synchronize():
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def make_candidates(count):
    return [f"candidate {index:02d}" for index in range(1, count + 1)]


def make_prefix(candidates):
    candidate_list = "\n".join(f"- {candidate}" for candidate in candidates)
    user_prompt = f"""STATE:
{state}

QUESTION:
{question}

CANDIDATES:
{candidate_list}

Return exactly one candidate."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def run_once(candidates, execution):
    synchronize()
    start = time.perf_counter()
    result = engine.choice(
        state=state,
        question=question,
        candidates=candidates,
        execution=execution,
    )
    synchronize()
    return (time.perf_counter() - start) * 1000.0, result


def stats(values):
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p95": statistics.quantiles(values, n=100, method="inclusive")[94],
        "min": min(values),
        "max": max(values),
    }


samples = {
    count: {
        "sequential": [],
        "batch": [],
        "speedup": [],
        "probability_delta": [],
        "selected_agreement": [],
    }
    for count in CANDIDATE_COUNTS
}

metadata = {}

for count in CANDIDATE_COUNTS:
    candidates = make_candidates(count)
    prefix = make_prefix(candidates)
    items = [
        tokenize_continuation(tokenizer, prefix, candidate)
        for candidate in candidates
    ]
    metadata[count] = {
        "prefix_tokens": len(tokenizer.encode(prefix, add_special_tokens=False)),
        "max_continuation_tokens": max(len(item.target_token_ids) for item in items),
    }


print("Warmup...")
for _ in range(WARMUP_ROUNDS):
    for count in CANDIDATE_COUNTS:
        candidates = make_candidates(count)
        run_once(candidates, "sequential")
        run_once(candidates, "batch")


round_orders = []

for round_index in range(ROUNDS):
    shift = round_index % len(CANDIDATE_COUNTS)
    order = CANDIDATE_COUNTS[shift:] + CANDIDATE_COUNTS[:shift]

    if round_index % 2 == 1:
        order = list(reversed(order))

    round_orders.append(order)
    print(f"Round {round_index + 1:02d}/{ROUNDS}: {order}")

    for position, count in enumerate(order):
        candidates = make_candidates(count)

        execution_order = (
            ["sequential", "batch"]
            if (round_index + position) % 2 == 0
            else ["batch", "sequential"]
        )

        timings = {}
        results = {}

        for execution in execution_order:
            elapsed, result = run_once(candidates, execution)
            timings[execution] = elapsed
            results[execution] = result

        samples[count]["sequential"].append(timings["sequential"])
        samples[count]["batch"].append(timings["batch"])
        samples[count]["speedup"].append(
            timings["sequential"] / timings["batch"]
        )

        delta = max(
            abs(
                results["sequential"].probabilities[candidate]
                - results["batch"].probabilities[candidate]
            )
            for candidate in candidates
        )

        samples[count]["probability_delta"].append(delta)
        samples[count]["selected_agreement"].append(
            results["sequential"].selected == results["batch"].selected
        )


rows = []

print("\nResults:")
for count in CANDIDATE_COUNTS:
    data = samples[count]
    row = {
        "candidate_count": count,
        "prefix_tokens": metadata[count]["prefix_tokens"],
        "max_continuation_tokens": metadata[count]["max_continuation_tokens"],
        "sequential_latency_ms": stats(data["sequential"]),
        "batch_latency_ms": stats(data["batch"]),
        "paired_speedup": stats(data["speedup"]),
        "max_probability_delta": max(data["probability_delta"]),
        "selected_agreement_rate": (
            sum(data["selected_agreement"]) / len(data["selected_agreement"])
        ),
    }
    rows.append(row)

    print(
        f"{count:>2} candidates | "
        f"prefix={row['prefix_tokens']:>3} tok | "
        f"seq={row['sequential_latency_ms']['median']:8.3f} ms | "
        f"batch={row['batch_latency_ms']['median']:8.3f} ms | "
        f"speedup={row['paired_speedup']['median']:6.3f}x"
    )

output = {
    "model": MODEL_ID,
    "device": str(device),
    "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    "dtype": str(next(engine.model.parameters()).dtype),
    "rounds": ROUNDS,
    "warmup_rounds": WARMUP_ROUNDS,
    "candidate_counts": CANDIDATE_COUNTS,
    "round_orders": round_orders,
    "results": rows,
}

Path("results").mkdir(exist_ok=True)
path = Path("results/candidate_scaling_benchmark.json")
path.write_text(json.dumps(output, indent=2), encoding="utf-8")

print()
print("Saved:", path)
