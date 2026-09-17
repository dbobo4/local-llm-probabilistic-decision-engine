# Local LLM Probabilistic Decision Engine

A local, open-source probabilistic decision engine built on top of open-weight LLMs. It replaces autoregressive answer generation with direct candidate scoring, producing structured, normalized decision probabilities with zero generated output tokens.

## Independent implementation

This is a fully independent implementation based only on publicly described product behavior. I have no access to TypeSafe/Jev internals, architecture, training code, weights, or proprietary technical details. The design in this repository is my own approach to building a local probabilistic decision engine with similar externally observable behavior.

This project does not reproduce, reverse-engineer, or claim knowledge of Jev's proprietary architecture. Any architectural choices in this repository are independent design decisions.

## What it does

Instead of asking a causal language model to autoregressively generate a textual answer, the engine directly scores the complete token sequence of each allowed candidate. The resulting candidate scores are normalized into a distribution over the supplied decision set.

Example input:

```text
State: My account has not worked for three days and I am losing sales.

Question: Which department should handle this?

Candidates:
- technical support
- billing
- sales
```

Measured result with `Qwen/Qwen2.5-1.5B-Instruct` using sequence log-likelihood scoring:

```text
technical support    94.65%
billing               0.64%
sales                 4.71%

Generated output tokens: 0
```

## Basic usage

```python
from llm_decision_engine import DecisionEngine

engine = DecisionEngine(
    model="Qwen/Qwen2.5-1.5B-Instruct",
)

result = engine.choice(
    state="My account has not worked for three days and I am losing sales.",
    question="Which department should handle this?",
    candidates=[
        "technical support",
        "billing",
        "sales",
    ],
)

print(result.probabilities)
```

Example result:

```python
{
    "technical support": 0.9465,
    "billing": 0.0064,
    "sales": 0.0471,
}
```

Additional result information includes the raw candidate sequence scores, selected candidate, scoring method, execution mode, candidate token counts, and generated output token count.

## Current implementation status

The repository contains a validated arbitrary multi-token candidate-scoring engine and an installable Python package. The original single-token A/B/C implementation is retained separately as a historical baseline experiment.

Current capabilities:

- Local open-weight Hugging Face causal language models
- Direct arbitrary multi-token candidate scoring
- Sequential reference execution with optional batched execution
- Sum log-likelihood scoring
- Optional mean log-likelihood scoring
- Normalized candidate-set probability distributions
- Zero generated output tokens for the decision
- Structured `ChoiceResult` output
- Tokenizer-boundary validation
- Correct causal-logit alignment
- Unit-tested scoring mathematics
- Installable `llm_decision_engine` Python package

Candidates are scored directly as their own token sequences. No intermediate A/B/C labels or 26-candidate limit are required.

Sequential execution is the default reference mode. Batched execution is available as an opt-in performance mode and evaluates all candidates in one model forward.

## Core idea

For a candidate token sequence

```math
c = (c_1, c_2, ..., c_T)
```

the default sequence score is

```math
S(c) = \sum_{t=1}^{T} \log P(c_t \mid x, c_1, \ldots, c_{t-1})
```

where `x` is the model context.

Candidate scores are then normalized across the allowed set `C`:

```math
P(c_i \mid x, c_i \in C) = \frac{\exp(S(c_i))}{\sum_j \exp(S(c_j))}
```

The optional `mean` scoring mode uses average token log-probability instead of the sum:

```math
S_{\mathrm{mean}}(c) = \frac{1}{T}\sum_{t=1}^{T} \log P(c_t \mid x, c_1, \ldots, c_{t-1})
```

The resulting values are normalized model preferences over the supplied candidate set. They should not be interpreted as calibrated probabilities of objective correctness unless calibration has been separately measured.

## Why causal alignment matters

For a prompt followed by candidate tokens `c1, c2`, a causal model predicts:

```text
last prompt position -> c1
c1 position          -> c2
```

The engine explicitly handles this one-token causal shift so that every candidate token is scored from the logits that actually predict it.

## Execution modes

The default execution mode is:

```python
execution="sequential"
```

Sequential mode performs one model forward per candidate and is treated as the numerical reference path.

Batched execution is available with:

```python
result = engine.choice(
    ...,
    execution="batch",
)
```

Batch mode evaluates all candidate continuations in one model forward. The scoring definition is unchanged, but exact numerical equivalence is not guaranteed under reduced-precision inference: GPU kernels and operation ordering can make BF16 results depend slightly on batch shape. For probability-sensitive evaluation, sequential mode remains the default reference.

## Boolean decisions

`boolean()` is a typed convenience primitive built on top of the same candidate-scoring path as `choice()`. It scores the continuations `yes` and `no` directly and returns probabilities plus a Python boolean decision. No answer tokens are generated.

~~~python
result = engine.boolean(
    state="The payment was charged twice.",
    question="Should this be escalated?",
)

print(result.probability_true)
print(result.probability_false)
print(result.selected)
~~~

The returned `BooleanResult` contains:

- `probability_true`: normalized probability assigned to `yes`
- `probability_false`: normalized probability assigned to `no`
- `selected`: `True` when `yes` has the higher candidate score, otherwise `False`
- `scores`: raw sequence log-likelihood scores for `yes` and `no`
- `scoring_method`: `sum` or `mean`
- `execution_mode`: `sequential` or `batch`
- `generated_output_tokens`: always `0` for this decision primitive

The probabilities are normalized model preferences over the two supplied semantic alternatives. They should not automatically be interpreted as calibrated probabilities of objective truth.

Batched execution is also supported:

~~~python
result = engine.boolean(
    state="The payment was charged twice.",
    question="Should this be escalated?",
    execution="batch",
)
~~~

A runnable example is available at `examples/basic_boolean.py`.

## Benchmark

A paired, interleaved execution benchmark was run with `Qwen/Qwen2.5-1.5B-Instruct` in BF16 on an NVIDIA GeForce RTX 5070 Ti using three candidate continuations.

~~~text
Sequential median latency:    54.551 ms
Batch median latency:         23.869 ms
Median paired speedup:         2.288x
Mean paired speedup:           2.262x
Sequential p95 latency:       98.988 ms
Batch p95 latency:            43.956 ms
Max probability delta:         0.002875
Selected candidate agreement: true
~~~

The benchmark alternates execution order between sequential and batch runs to reduce time-dependent GPU and system effects. These measurements are hardware-, model-, prompt-, candidate-set-, and precision-specific and should not be interpreted as universal performance guarantees.

Under BF16, sequential and batched execution can produce slightly different numerical probabilities even though they implement the same candidate-scoring definition. FP32 control experiments showed near-equivalence, indicating that the observed drift is primarily a reduced-precision numerical effect.

The reproducible benchmark is available in `benchmarks/execution_modes_paired.py`.

## Engine roadmap

```text
Implemented
- Choice
- arbitrary multi-token candidates
- sum and mean sequence scoring
- sequential and batched execution modes
- tokenizer-boundary validation
- Hugging Face / PyTorch backend

Planned
- Boolean / proposition probability
- Score
- multiple questions per state
- probability calibration
- llama.cpp / GGUF backend
```

## Research

The project is also a reproducible research platform for studying when autoregressive generation is actually necessary for decision tasks.

The central research question is:

> When can autoregressive generation be replaced by direct probabilistic decision inference?

Planned comparison:

```text
A) Autoregressive reasoning + answer
B) Direct candidate scoring
C) Extra compute + candidate scoring
```

Primary measurements:

- accuracy vs. latency
- calibration quality
- candidate wording sensitivity
- candidate ordering sensitivity
- candidate-cardinality scaling
- compute / quality trade-offs

The goal is to distinguish the value of generated reasoning tokens from the value of additional computation itself.

## Repository structure

```text
src/llm_decision_engine/   reusable library
examples/                  user-facing examples
experiments/               research and historical baselines
benchmarks/                performance and quality benchmarks
tests/                     automated tests
results/                   reproducible published results
```

## Development install

```text
python -m pip install -e ".[dev]"
```

Run the example:

```text
python examples/basic_choice.py
```

Run the tests:

```text
python -m pytest -q
```

Build the package:

```text
python -m build
```

## Model weights

Model weights are not distributed with this repository. Models are downloaded from their original providers and cached locally by the selected inference backend. Inference itself runs locally on the user's machine. Users are responsible for complying with the license and terms of the model they choose.

## Project status

Early development. Arbitrary multi-token candidate scoring and batched execution are implemented and validated. Calibration, benchmarking, additional decision primitives, and backend support remain under active development.
