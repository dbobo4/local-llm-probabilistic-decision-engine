import json
import statistics
import time
from pathlib import Path

import torch

from llm_decision_engine import DecisionEngine


MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
PAIRS = 30
WARMUP = 3

state = (
    "The customer account has not worked for three days "
    "and they are losing sales."
)
question = "Which department should handle this?"
candidates = ["technical support", "billing", "sales"]

engine = DecisionEngine(model=MODEL_ID)
device = next(engine.model.parameters()).device


def synchronize():
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_once(execution):
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


for _ in range(WARMUP):
    run_once("sequential")
    run_once("batch")

sequential_ms = []
batch_ms = []
paired_speedups = []
last_sequential = None
last_batch = None

for pair_index in range(PAIRS):
    order = (
        ["sequential", "batch"]
        if pair_index % 2 == 0
        else ["batch", "sequential"]
    )

    timings = {}

    for execution in order:
        elapsed_ms, result = run_once(execution)
        timings[execution] = elapsed_ms

        if execution == "sequential":
            last_sequential = result
        else:
            last_batch = result

    sequential_ms.append(timings["sequential"])
    batch_ms.append(timings["batch"])
    paired_speedups.append(
        timings["sequential"] / timings["batch"]
    )


def stats(values):
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p95": statistics.quantiles(
            values, n=100, method="inclusive"
        )[94],
        "min": min(values),
        "max": max(values),
    }


probability_delta = {
    candidate: abs(
        last_sequential.probabilities[candidate]
        - last_batch.probabilities[candidate]
    )
    for candidate in candidates
}

output = {
    "model": MODEL_ID,
    "pairs": PAIRS,
    "warmup": WARMUP,
    "device": str(device),
    "gpu": (
        torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else None
    ),
    "dtype": str(next(engine.model.parameters()).dtype),
    "sequential_latency_ms": stats(sequential_ms),
    "batch_latency_ms": stats(batch_ms),
    "paired_speedup": stats(paired_speedups),
    "max_probability_delta": max(probability_delta.values()),
    "probability_delta": probability_delta,
    "selected_candidate_agreement": (
        last_sequential.selected == last_batch.selected
    ),
}

Path("results").mkdir(exist_ok=True)
path = Path("results/execution_mode_paired_benchmark.json")
path.write_text(json.dumps(output, indent=2), encoding="utf-8")

print("Sequential median:      " f"{statistics.median(sequential_ms):.3f} ms")
print("Batch median:           " f"{statistics.median(batch_ms):.3f} ms")
print("Median paired speedup:  " f"{statistics.median(paired_speedups):.3f}x")
print("Mean paired speedup:    " f"{statistics.mean(paired_speedups):.3f}x")
print("P95 sequential:         " f"{output['sequential_latency_ms']['p95']:.3f} ms")
print("P95 batch:              " f"{output['batch_latency_ms']['p95']:.3f} ms")
print("Max probability delta: " f"{output['max_probability_delta']:.9f}")
print("Selected agreement:     " f"{output['selected_candidate_agreement']}")
print()
print("Saved:", path)
