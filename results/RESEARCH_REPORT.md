# Research Report

This report summarizes committed research results for the Local LLM Probabilistic Decision Engine. It is generated from committed result JSON files and does not run model inference.

## Research question

> When can autoregressive generation be replaced by direct probabilistic decision inference?

The main method scores complete candidate continuations directly under a causal language model. It does not autoregressively generate an answer.

The experiments below are intended to identify both useful operating regions and failure boundaries. They do not establish universal equivalence between direct scoring and generation.

## Main routing result

The routing evaluation used a pre-specified, frozen protocol with six support categories and explicit category definitions. The test split was held out from model evaluation until the frozen run.

- Evaluation type: `frozen_pre_specified_held_out_test`
- Dataset SHA256: `abf06b95a126d122822a5cb17e0141d009e59d24de58e01878fcd834300cc6ee`
- Direct scoring: sum sequence log-likelihood, sequential execution
- Direct generated output tokens: 0
- Greedy control: deterministic generation with exact canonical-candidate parsing

| Model | Direct accuracy | Top-2 | Greedy accuracy | Direct/greedy agreement | Direct output tokens | Greedy output tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Qwen/Qwen2.5-1.5B-Instruct` | 88.3% | 100.0% | 87.5% | 99.2% | 0 | 304 |
| `Qwen/Qwen2.5-3B-Instruct` | 96.7% | 100.0% | 96.7% | 100.0% | 0 | 300 |

On this controlled routing task, direct candidate scoring reproduced the greedy route decision almost perfectly for the 1.5B model and exactly for the 3B model, while the direct method produced no autoregressively generated output tokens.

This result is specific to this benchmark, candidate schema, prompt protocol, and tested models. It is not a general proof that direct scoring and greedy generation are equivalent.

This was not a blind benchmark: the benchmark generator and template source were known during benchmark construction. The test split was nevertheless frozen before model evaluation. It has now been evaluated and is therefore an opened test set; it should not be used for further tuning while still being described as untouched held-out evidence.

## Effect of explicit candidate definitions

Before the frozen test, calibration experiments compared bare candidate labels with the same labels accompanied by fixed semantic definitions.

| Model | Bare accuracy | Definition-guided accuracy | Decisions changed | Errors fixed | Correct decisions broken |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Qwen/Qwen2.5-1.5B-Instruct` | 70.8% | 90.0% | 25 | 24 | 1 |
| `Qwen/Qwen2.5-3B-Instruct` | 75.0% | 100.0% | 30 | 30 | 0 |

The calibration result supports an important design principle for the public API: candidate descriptions can provide an explicit decision schema while the candidate key itself remains the scored output.

This was a calibration-stage result on a synthetic routing benchmark, not an independent real-world routing evaluation.

## Candidate-order sensitivity

Candidate order is part of the prompt seen by the model. Reordering the same six candidates changed some decisions, especially for the smaller model.

| Model | Canonical accuracy | Reversed | Rotated | Fixed shuffle | Stable across all tested orders |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Qwen/Qwen2.5-1.5B-Instruct` | 90.0% | 97.5% | 94.2% | 91.7% | 99/120 (82.5%) |
| `Qwen/Qwen2.5-3B-Instruct` | 100.0% | 100.0% | 99.2% | 98.3% | 117/120 (97.5%) |

The library therefore preserves user-supplied insertion order rather than silently sorting candidates. Applications that depend on stable behavior should keep candidate ordering deterministic and evaluate order sensitivity on their own task.

## Sum versus mean scoring

The default method uses the full candidate sequence log-likelihood:

```math
S(c) = \sum_t \log P(c_t \mid x, c_{<t})
```

A length-normalized mean-log-probability alternative was also tested.

| Model | Sum accuracy | Mean accuracy | Decision switches |
| --- | ---: | ---: | ---: |
| `Qwen/Qwen2.5-1.5B-Instruct` | 70.8% | 70.8% | 0 |
| `Qwen/Qwen2.5-3B-Instruct` | 75.0% | 75.8% | 1 |

Mean scoring did not materially change the smaller model's decisions and changed only one decision for the 3B model in this calibration experiment. The result does not justify replacing sum scoring as the default.

Sum remains the principled default because it corresponds to the probability of the complete candidate token sequence in log-space. Mean scoring is exposed as an alternative diagnostic.

## Boundary condition: binary arithmetic and verification tasks

A separate pre-specified 120-example test, held out from model evaluation until the frozen run, tested the dedicated direct True/False protocol on arithmetic and verification-style tasks.

- Evaluation type: `held_out_test`
- Protocol frozen before test: `true`
- Dataset SHA256: `50ffd1d786995835c22e8e35d84545b7d76dec167ed3e259b87876ac8885e749`
- Main method: direct dedicated True/False probabilistic scoring
- Main-method generated output tokens: 0
- Diagnostic control: structured verification generation followed by direct scoring

| Model | Direct accuracy | Direct Brier | Direct NLL | Reasoning-conditioned accuracy |
| --- | ---: | ---: | ---: | ---: |
| `Qwen/Qwen2.5-1.5B-Instruct` | 65.0% | 0.276 | 0.892 | 94.2% |
| `Qwen/Qwen2.5-3B-Instruct` | 74.2% | 0.240 | 3.202 | 85.0% |

This benchmark is important negative evidence. Direct answer readout can fail when the task requires intermediate computation that is not already represented strongly enough at the answer position.

The reasoning-conditioned control generated intermediate verification text before performing the final direct readout. Its improvement shows that additional computation can matter, but that control is outside the main no-generation method.

The project therefore does not claim that direct scoring should replace generation on reasoning-heavy tasks.

## What the evidence currently supports

- On the tested routing task, direct scoring worked well for constrained semantic decisions where the allowed outputs and decision schema were known in advance.
- Explicit candidate definitions can substantially improve decisions when short labels are semantically ambiguous.
- Multi-token candidates can be scored directly as complete causal continuations.
- Direct candidate scoring can closely reproduce greedy output decisions on the tested routing task without generating answer tokens.
- Candidate ordering can influence results and should be treated as part of the inference protocol.
- Direct answer readout can be substantially weaker when intermediate computation is required.

## What the evidence does not establish

- It does not prove universal equivalence between direct scoring and autoregressive generation.
- It does not show that direct scoring is always faster or uses less total compute.
- It does not establish that normalized candidate probabilities are calibrated real-world probabilities of correctness.
- It does not establish production routing quality outside the synthetic benchmark.
- It does not establish compatibility or equivalent behavior across all causal language models.

## Probability interpretation

The engine normalizes scores across the supplied candidate set. The resulting values are restricted-choice model probabilities over those candidates.

A high normalized probability should not automatically be interpreted as a calibrated probability that the selected candidate is objectively correct. Calibration must be measured separately on representative labeled data.

## Performance benchmarks

Sequential-versus-batch latency and candidate-count scaling are reported separately in `results/BENCHMARK_REPORT.md`.

Those performance measurements are hardware-, model-, precision-, prompt-, and candidate-set-specific and are not universal speed guarantees.

## Reproduction

The repository retains the benchmark generators, frozen datasets, evaluation scripts, result JSON files, dataset hashes, and relevant protocol metadata needed to inspect and reproduce the experiments.

Key result files:

- `results/routing_v1_frozen_test.json`
- `results/routing_v1_definition_guided_calibration.json`
- `results/routing_v1_candidate_order_calibration.json`
- `results/routing_v1_sum_vs_mean_calibration.json`
- `results/binary_quality_v2_test_final.json`
- `results/BENCHMARK_REPORT.md`
