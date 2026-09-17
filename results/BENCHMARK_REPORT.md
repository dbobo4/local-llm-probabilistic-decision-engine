# Benchmark Report

This report is generated from committed benchmark result files. It does not run model inference.

## Environment

- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- GPU: `NVIDIA GeForce RTX 5070 Ti`
- Device: `cuda:0`
- Precision: `torch.bfloat16`

## Paired execution benchmark

Sequential and batched execution are measured in alternating order to reduce time-dependent GPU and system effects.

| Metric | Sequential | Batch |
| --- | ---: | ---: |
| Median latency | 54.551 ms | 23.869 ms |
| P95 latency | 98.988 ms | 43.956 ms |
| Mean latency | 61.535 ms | 28.022 ms |

- Median paired speedup: **2.288x**
- Mean paired speedup: **2.262x**
- Max absolute probability delta: **0.002875**
- Selected candidate agreement: **true**

## Candidate-count scaling

The scaling benchmark interleaves candidate counts across rounds and alternates sequential/batch execution order.

| Candidates | Prefix tokens | Sequential median | Batch median | Median speedup | Max abs probability delta | Agreement |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 73 | 36.187 ms | 19.514 ms | 1.808x | 0.000000 | 100% |
| 3 | 79 | 53.495 ms | 19.485 ms | 2.731x | 0.011288 | 100% |
| 5 | 91 | 86.646 ms | 27.948 ms | 3.124x | 0.000000 | 100% |
| 10 | 121 | 173.094 ms | 59.570 ms | 2.931x | 0.023715 | 100% |
| 20 | 181 | 375.603 ms | 162.342 ms | 2.277x | 0.010669 | 100% |

## Interpretation

Sequential execution remains the numerical reference path. Batched execution reduces latency by evaluating candidate continuations together, but BF16 execution can introduce batch-shape-dependent numerical differences in the resulting normalized probabilities.

The measured speedups are specific to the model, GPU, prompt structure, candidate set, precision, software stack, and benchmark procedure used here. They are not universal performance guarantees.

## Reproduction

- `benchmarks/execution_modes_paired.py` produces the paired execution result.
- `benchmarks/candidate_scaling.py` produces the candidate-scaling result.
- `benchmarks/generate_report.py` generates this report from those JSON files.
