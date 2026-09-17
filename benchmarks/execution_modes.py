import json
import statistics
import subprocess
import time
from pathlib import Path

import torch

from llm_decision_engine import DecisionEngine


MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
REPEATS = 30

state = (
    "The customer account has not worked for three days "
    "and they are losing sales."
)
question = "Which department should handle this?"
candidates = ["technical support", "billing", "sales"]


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return None


engine = DecisionEngine(model=MODEL_ID)
device = next(engine.model.parameters()).device
dtype = str(next(engine.model.parameters()).dtype)


def synchronize():
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_mode(execution):
    engine.choice(
        state=state,
        question=question,
        candidates=candidates,
        execution=execution,
    )
    synchronize()

    latencies_ms = []
    incremental_peak_mb = []
    last_result = None

    for _ in range(REPEATS):
        if device.type == "cuda":
            baseline = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
        else:
            baseline = 0

        synchronize()
        start = time.perf_counter()

        last_result = engine.choice(
            state=state,
            question=question,
            candidates=candidates,
            execution=execution,
        )

        synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        latencies_ms.append(elapsed_ms)

        if device.type == "cuda":
            peak = torch.cuda.max_memory_allocated(device)
            incremental_peak_mb.append(
                max(0, peak - baseline) / (1024 ** 2)
            )

    return {
        "execution": execution,
        "latency_ms": {
            "mean": statistics.mean(latencies_ms),
            "median": statistics.median(latencies_ms),
            "p95": statistics.quantiles(latencies_ms, n=100, method="inclusive")[94],
            "min": min(latencies_ms),
            "max": max(latencies_ms),
            "runs": latencies_ms,
        },
        "incremental_peak_memory_mb": (
            max(incremental_peak_mb)
            if incremental_peak_mb
            else None
        ),
        "probabilities": last_result.probabilities,
        "scores": last_result.scores,
        "selected": last_result.selected,
        "token_counts": last_result.token_counts,
    }


sequential = benchmark_mode("sequential")
batched = benchmark_mode("batch")

probability_delta = {
    candidate: abs(
        sequential["probabilities"][candidate]
        - batched["probabilities"][candidate]
    )
    for candidate in candidates
}

comparison = {
    "speedup": (
        sequential["latency_ms"]["mean"]
        / batched["latency_ms"]["mean"]
    ),
    "probability_delta": probability_delta,
    "max_probability_delta": max(probability_delta.values()),
    "selected_candidate_agreement": (
        sequential["selected"] == batched["selected"]
    ),
}

gpu_name = None
if device.type == "cuda":
    gpu_name = torch.cuda.get_device_name(device)

output = {
    "model": MODEL_ID,
    "git_commit": git_commit(),
    "torch_version": torch.__version__,
    "device": str(device),
    "gpu": gpu_name,
    "dtype": dtype,
    "scoring": "sum",
    "repeats": REPEATS,
    "candidates": candidates,
    "sequential": sequential,
    "batch": batched,
    "comparison": comparison,
}

Path("results").mkdir(exist_ok=True)
output_path = Path("results/execution_mode_benchmark.json")
output_path.write_text(
    json.dumps(output, indent=2),
    encoding="utf-8",
)

print("Sequential mean latency: " f"{sequential['latency_ms']['mean']:.3f} ms")
print("Batch mean latency:      " f"{batched['latency_ms']['mean']:.3f} ms")
print("Speedup:                 " f"{comparison['speedup']:.3f}x")
print("Sequential peak extra:   " f"{sequential['incremental_peak_memory_mb']:.2f} MB")
print("Batch peak extra:        " f"{batched['incremental_peak_memory_mb']:.2f} MB")
print("Max probability delta:   " f"{comparison['max_probability_delta']:.9f}")
print("Selected agreement:      " f"{comparison['selected_candidate_agreement']}")
print()
print("Sequential probabilities:", sequential["probabilities"])
print("Batch probabilities:     ", batched["probabilities"])
print()
print("Saved:", output_path)
