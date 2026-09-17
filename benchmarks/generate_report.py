import json
from pathlib import Path


RESULTS_DIR = Path("results")
PAIRED_PATH = RESULTS_DIR / "execution_mode_paired_benchmark.json"
SCALING_PATH = RESULTS_DIR / "candidate_scaling_benchmark.json"
OUTPUT_PATH = RESULTS_DIR / "BENCHMARK_REPORT.md"


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


paired = load_json(PAIRED_PATH)
scaling = load_json(SCALING_PATH)

lines = [
    "# Benchmark Report",
    "",
    "This report is generated from committed benchmark result files. It does not run model inference.",
    "",
    "## Environment",
    "",
    f"- Model: `{paired['model']}`",
    f"- GPU: `{paired['gpu']}`",
    f"- Device: `{paired['device']}`",
    f"- Precision: `{paired['dtype']}`",
    "",
    "## Paired execution benchmark",
    "",
    "Sequential and batched execution are measured in alternating order to reduce time-dependent GPU and system effects.",
    "",
    "| Metric | Sequential | Batch |",
    "| --- | ---: | ---: |",
    f"| Median latency | {paired['sequential_latency_ms']['median']:.3f} ms | {paired['batch_latency_ms']['median']:.3f} ms |",
    f"| P95 latency | {paired['sequential_latency_ms']['p95']:.3f} ms | {paired['batch_latency_ms']['p95']:.3f} ms |",
    f"| Mean latency | {paired['sequential_latency_ms']['mean']:.3f} ms | {paired['batch_latency_ms']['mean']:.3f} ms |",
    "",
    f"- Median paired speedup: **{paired['paired_speedup']['median']:.3f}x**",
    f"- Mean paired speedup: **{paired['paired_speedup']['mean']:.3f}x**",
    f"- Max absolute probability delta: **{paired['max_probability_delta']:.6f}**",
    f"- Selected candidate agreement: **{str(paired['selected_candidate_agreement']).lower()}**",
    "",
    "## Candidate-count scaling",
    "",
    "The scaling benchmark interleaves candidate counts across rounds and alternates sequential/batch execution order.",
    "",
    "| Candidates | Prefix tokens | Sequential median | Batch median | Median speedup | Max abs probability delta | Agreement |",
    "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
]

for row in scaling["results"]:
    lines.append(
        f"| {row['candidate_count']} "
        f"| {row['prefix_tokens']} "
        f"| {row['sequential_latency_ms']['median']:.3f} ms "
        f"| {row['batch_latency_ms']['median']:.3f} ms "
        f"| {row['paired_speedup']['median']:.3f}x "
        f"| {row['max_probability_delta']:.6f} "
        f"| {row['selected_agreement_rate'] * 100:.0f}% |"
    )

lines += [
    "",
    "## Interpretation",
    "",
    "Sequential execution remains the numerical reference path. Batched execution reduces latency by evaluating candidate continuations together, but BF16 execution can introduce batch-shape-dependent numerical differences in the resulting normalized probabilities.",
    "",
    "The measured speedups are specific to the model, GPU, prompt structure, candidate set, precision, software stack, and benchmark procedure used here. They are not universal performance guarantees.",
    "",
    "## Reproduction",
    "",
    "- `benchmarks/execution_modes_paired.py` produces the paired execution result.",
    "- `benchmarks/candidate_scaling.py` produces the candidate-scaling result.",
    "- `benchmarks/generate_report.py` generates this report from those JSON files.",
    "",
]

OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")
print("Generated:", OUTPUT_PATH)
