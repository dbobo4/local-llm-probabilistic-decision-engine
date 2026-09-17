# Local LLM Probabilistic Decision Engine

A local, open-source Python library for turning compatible causal language models into structured probabilistic decision engines through direct candidate-sequence scoring.

Instead of asking the model to autoregressively generate an answer, the library scores the allowed answers directly and returns normalized probabilities over the supplied decision set.

```python
from llm_decision_engine import DecisionEngine

engine = DecisionEngine(
    model="Qwen/Qwen2.5-1.5B-Instruct",
)

result = engine.choice(
    state="The customer cannot sign in after resetting their password twice.",
    question="Which support category best matches this request?",
    candidates={
        "billing": "Charges, invoices, payments, receipts, refunds, and unexpected fees.",
        "technical support": "Product failures, crashes, errors, broken features, or synchronization problems.",
        "account access": "Login, password, authentication, locked-account, or two-factor-access problems.",
    },
)

print(result.selected)
print(result.probabilities)
print(result.generated_output_tokens)
```

The main decision methods produce **zero autoregressively generated output tokens**.

## Why this exists

Many LLM tasks are not fundamentally open-ended generation tasks.

Sometimes the application already knows the allowed outputs:

- route a support request to one of several teams
- decide whether a condition is true or false
- assign a severity level
- classify an event into a known schema
- choose a workflow branch
- evaluate several structured questions over the same state

For these cases, generating free-form text and parsing it back into a structured decision may be unnecessary.

This project explores a different inference pattern:

```text
state + question + allowed candidates
                |
                v
       causal language model
                |
                v
   candidate sequence likelihoods
                |
                v
 normalized decision probabilities
```

The central research question is:

> When can autoregressive generation be replaced by direct probabilistic decision inference?

## Installation

### From PyPI

After the package is published on PyPI, installation will be:

```text
pip install llm-decision-engine
```

The Python import will remain:

```python
import llm_decision_engine
```

### From source

```text
git clone https://github.com/dbobo4/local-llm-probabilistic-decision-engine.git
cd local-llm-probabilistic-decision-engine
python -m pip install .
```

For development:

```text
python -m pip install -e ".[dev]"
```

## Model loading

A model must be supplied explicitly.

### Hugging Face model ID

```python
from llm_decision_engine import DecisionEngine

engine = DecisionEngine(
    model="Qwen/Qwen2.5-1.5B-Instruct",
)
```

If the required files are not already cached, Hugging Face Transformers may download them from the model provider.

### Local model directory

```python
engine = DecisionEngine(
    model="./models/Qwen2.5-1.5B-Instruct",
)
```

### Offline-only loading

```python
engine = DecisionEngine(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    local_files_only=True,
)
```

With `local_files_only=True`, the required model files must already exist locally, either in the Hugging Face cache or in the directory supplied through `model=`.

Model weights are not bundled with this package.

See [MODEL_COMPATIBILITY.md](https://github.com/dbobo4/local-llm-probabilistic-decision-engine/blob/main/MODEL_COMPATIBILITY.md) for model-related notes.

## Core API

The v0.1.0 decision API contains three primitives:

```text
choice()
boolean()
rating()
```

and one orchestration method:

```text
decide()
```

`decide()` is not a fourth inference primitive. It composes the existing decision types over a shared state.

## Choice

Use `choice()` when the valid outputs are known in advance.

```python
result = engine.choice(
    state="The customer was charged twice for the same invoice.",
    question="Which team should handle this?",
    candidates=[
        "billing",
        "technical support",
        "sales",
    ],
)

print(result.selected)
print(result.probabilities)
```

Candidates may contain multiple tokens. They are scored as complete causal continuations rather than being reduced to single-token labels.

The result includes:

```text
probabilities
scores
selected
token_counts
scoring_method
execution_mode
generated_output_tokens
```

## Described choices

Short candidate names can be ambiguous. `choice()` can therefore accept a dictionary mapping each scored candidate to a semantic description.

```python
result = engine.choice(
    state="The customer cannot sign in after resetting their password.",
    question="Which support category best matches this request?",
    candidates={
        "billing": "Charges, invoices, payments, receipts, refunds, and unexpected fees.",
        "technical support": "Product failures, crashes, errors, broken features, or synchronization problems.",
        "account access": "Login, password, authentication, locked-account, or two-factor-access problems.",
    },
)
```

Only the dictionary keys are scored as candidate continuations.

The descriptions are added to the model context to define what each candidate means.

The returned result therefore remains keyed by:

```text
billing
technical support
account access
```

rather than by the longer descriptions.

Candidate insertion order is preserved. This matters because language-model decisions can be sensitive to candidate ordering. Applications should keep candidate order deterministic and evaluate order sensitivity on representative data.

Runnable example:

```text
examples/described_choice.py
```

## Boolean decisions

`boolean()` uses a dedicated binary protocol with the fixed candidate continuations `True` and `False`.

```python
result = engine.boolean(
    state="The payment was charged twice.",
    question="Should this be escalated?",
)

print(result.probability_true)
print(result.probability_false)
print(result.selected)
```

`selected` is a Python `bool`.

The result includes:

```text
probability_true
probability_false
selected
scores
scoring_method
execution_mode
generated_output_tokens
```

Runnable example:

```text
examples/basic_boolean.py
```

## Numeric ratings

`rating()` can score an arbitrary ordered set of integer levels.

```python
result = engine.rating(
    state="The response is mostly correct but contains one minor factual error.",
    question="Rate the reliability from 1 to 5.",
    levels=[1, 2, 3, 4, 5],
)

print(result.probabilities)
print(result.selected)
print(result.expected_value)
```

For numeric levels, the expected value is:

```text
E[R] = sum(level_i * P(level_i))
```

`selected` is the highest-probability level.

`expected_value` uses the full distribution and therefore does not need to equal `selected`.

Runnable example:

```text
examples/basic_rating.py
```

## Named rating levels

Ratings are not restricted to numeric scales.

```python
result = engine.rating(
    state="The issue blocks the customer from using a core product feature.",
    question="How severe is this issue?",
    levels=[
        "minor",
        "moderate",
        "serious",
        "critical",
    ],
)

print(result.probabilities)
print(result.selected)
print(result.expected_value)
```

For string levels, `expected_value` is `None` because the library does not invent a numeric distance between user-defined labels.

Runnable example:

```text
examples/text_rating.py
```

## Multiple structured decisions

`decide()` applies several decision specifications to the same state.

```python
from llm_decision_engine import Boolean, Choice, DecisionEngine, Rating

engine = DecisionEngine(
    model="Qwen/Qwen2.5-1.5B-Instruct",
)

message = "I cannot log in and I need access immediately."

result = engine.decide(
    state=message,
    questions={
        "route": Choice(
            question="Which team should handle this request?",
            candidates={
                "billing": "Charges, invoices, payments, receipts, refunds, and unexpected fees.",
                "technical support": "Product failures, crashes, errors, broken features, or synchronization problems.",
                "account access": "Login, password, authentication, locked-account, or two-factor-access problems.",
            },
        ),
        "urgent": Boolean(
            question="Does this request require urgent attention?",
        ),
        "severity": Rating(
            question="How severe is the issue?",
            levels=[
                "minor",
                "moderate",
                "serious",
                "critical",
            ],
        ),
    },
)

print(result.results["route"].selected)
print(result.results["urgent"].selected)
print(result.results["severity"].selected)
```

Question insertion order is preserved.

Each specification may independently choose:

```python
scoring="sum"        # or "mean"
execution="sequential"  # or "batch"
```

`decide()` currently delegates to the individual primitives. It does not fuse all questions into a single model forward.

Runnable example:

```text
examples/decide.py
```

## Direct sequence scoring

For a candidate token sequence

```math
c=(c_1,c_2,\dots,c_T)
```

the default sequence score is

```math
S(c)=\sum_{t=1}^{T}\log P(c_t \mid x,c_1,\dots,c_{t-1})
```

where $x$ is the complete model context.

In probability space, this corresponds to the probability of the complete candidate continuation:

```math
P(c \mid x)=\prod_{t=1}^{T}P(c_t \mid x,c_1,\dots,c_{t-1})
```

Written out explicitly:

```math
P(c \mid x)=P(c_1 \mid x)\cdot P(c_2 \mid x,c_1)\cdot\dots\cdot P(c_T \mid x,c_1,\dots,c_{T-1})
```

The implementation works in log-space because adding log-probabilities is numerically more stable than multiplying many small probabilities.

Candidate scores are then normalized across the supplied candidate set $C$:

```math
P(c_i \mid x,\;c_i\in C)=\frac{\exp(S(c_i))}{\sum_j\exp(S(c_j))}
```

This produces a probability distribution over the supplied candidate set.

The optional `mean` scoring mode uses average token log-probability instead of the sum:

```math
S_{\mathrm{mean}}(c)=\frac{1}{T}\sum_{t=1}^{T}\log P(c_t \mid x,c_1,\dots,c_{t-1})
```

The resulting normalized values represent model preferences over the supplied candidates. They should not automatically be interpreted as calibrated probabilities of objective correctness unless calibration has been evaluated separately.

## Probability interpretation

The returned probabilities are **restricted-candidate model probabilities**.

For example:

```python
{
    "billing": 0.82,
    "technical support": 0.11,
    "sales": 0.07,
}
```

means that the model assigns 82% of the normalized probability mass to `billing` among those supplied candidates under the current inference protocol.

It does **not** automatically mean:

```text
There is an objectively calibrated 82% probability that billing is correct.
```

Calibration must be measured separately on representative labeled data.

A candidate absent from the supplied set cannot receive probability mass.

## Sum versus mean scoring

The default is:

```python
scoring="sum"
```

This uses the log-likelihood of the complete candidate sequence:

```text
sum(log token probabilities)
```

It is the direct log-space form of the full candidate sequence probability.

An alternative is:

```python
scoring="mean"
```

which uses:

```text
mean(log token probabilities)
```

This length-normalized score can be useful as a diagnostic when candidate lengths differ, but it is not the same probabilistic quantity as the full sequence probability.

`sum` therefore remains the default.

## Sequential versus batch execution

The default execution mode is:

```python
execution="sequential"
```

Sequential mode evaluates candidate continuations separately and is treated as the numerical reference path.

Batch mode is available through:

```python
execution="batch"
```

and evaluates the candidate continuations together in one model forward.

Example:

```python
result = engine.choice(
    state="The parcel has not arrived.",
    question="Which team should handle this?",
    candidates=[
        "billing",
        "technical support",
        "shipping",
    ],
    execution="batch",
)
```

Both modes implement the same scoring definition, but reduced-precision inference can produce small batch-shape-dependent numerical differences.

For this reason, bit-identical probabilities between sequential and batch execution are not guaranteed.

## What zero generated output tokens means

The main inference path does not autoregressively generate the selected answer.

It evaluates model logits and directly scores the allowed continuations.

Therefore:

```text
generated_output_tokens == 0
```

for the direct decision methods.

This does **not** mean that no model computation occurs.

The causal language model still performs forward inference to produce logits, and direct scoring is not automatically guaranteed to use less total compute or lower latency than every generation-based alternative.

## Important limitation: direct scoring does not add reasoning steps

Direct candidate scoring reads the model's probabilities over the allowed continuations without first generating an intermediate reasoning trace.

This means that the method depends strongly on what the underlying model can already represent at the answer position. A model may sometimes assign more probability to the wrong candidate even on a seemingly simple verification task, despite being able to reach the correct answer when allowed to generate intermediate reasoning.

A wrong direct decision is therefore not, by itself, evidence of a scoring implementation error. When the causal sequence likelihood is computed correctly, the engine is faithfully exposing the model's conditional preference over the supplied candidates. That model preference can still be wrong.

The binary arithmetic and verification benchmark in this repository demonstrates this boundary. Reasoning-conditioned evaluation substantially improved accuracy, but it generated intermediate verification text and therefore falls outside the library's main zero-generated-output-token method.

For reasoning-heavy tasks, the main ways to improve reliability are:

- use a stronger base model whose reasoning capability is already reflected more reliably in the direct candidate logits
- improve the prompt and candidate definitions when ambiguity or task framing is the main problem
- allow generated intermediate reasoning when the application values reasoning quality more than the zero-generated-output-token constraint

The last option changes the inference method: once intermediate reasoning is generated, the system is no longer using the library's main zero-output-token decision path.

If zero generated output tokens must be preserved, stronger base-model capability is therefore especially important for tasks that require computation or multi-step reasoning.

In short:

**Stronger base-model capability -> more reliable direct decision readout.**

**Generated reasoning -> potentially better reasoning, but no longer zero output tokens.**

## Example applications

The API is designed for tasks where the output space is known in advance, such as:

- customer-support routing
- intent classification
- moderation categories
- workflow branching
- binary policy checks
- risk or severity levels
- reliability or quality ratings
- event classification
- tool or handler selection
- structured decision extraction from text
- several related decisions over the same state

For example, an application can turn one customer message into:

```text
route    -> account access
urgent   -> True
severity -> serious
```

while retaining a probability distribution for every decision.

## When this approach is a good fit

Direct candidate scoring is most natural when:

- the allowed outputs are known before inference
- the task is primarily a constrained semantic decision
- structured output is required
- free-form generation is unnecessary
- candidate probabilities are useful to downstream logic
- deterministic output schemas are valuable

The routing experiments in this repository provide controlled evidence that this can work well for that type of problem.

## When not to use it

Direct answer scoring should not be assumed to replace generation when:

- the answer space is open-ended
- the correct answer may not be present in the supplied candidates
- the task requires substantial intermediate reasoning or computation
- creative or explanatory text is required
- the application needs the model to construct a new answer rather than select among known alternatives

The project's binary arithmetic and verification benchmark provides an explicit negative result: direct answer readout was substantially weaker than a reasoning-conditioned control on tasks requiring intermediate computation.

## Research results

The repository intentionally includes both positive and negative results.

### Frozen routing evaluation

A six-class synthetic support-routing benchmark used a pre-specified protocol frozen before model evaluation.

| Model | Direct accuracy | Greedy accuracy | Direct / greedy agreement | Direct generated tokens |
| --- | ---: | ---: | ---: | ---: |
| Qwen2.5-1.5B-Instruct | 88.3% | 87.5% | 99.2% | 0 |
| Qwen2.5-3B-Instruct | 96.7% | 96.7% | 100.0% | 0 |

The test split was held out from model evaluation until the frozen run, but this was not a blind benchmark: the benchmark generator and template source were known during benchmark construction.

The result is specific to this synthetic task, prompt protocol, candidate schema, and the two tested models. It is not evidence of universal equivalence between direct scoring and generation.

### Explicit candidate definitions

On the routing calibration split:

| Model | Bare labels | Definition-guided |
| --- | ---: | ---: |
| Qwen2.5-1.5B-Instruct | 70.8% | 90.0% |
| Qwen2.5-3B-Instruct | 75.0% | 100.0% |

This experiment motivated support for described candidates in the public API.

### Reasoning boundary

On a separate 120-example binary arithmetic and verification test:

| Model | Direct accuracy | Reasoning-conditioned accuracy |
| --- | ---: | ---: |
| Qwen2.5-1.5B-Instruct | 65.0% | 94.2% |
| Qwen2.5-3B-Instruct | 74.2% | 85.0% |

The reasoning-conditioned path generated intermediate verification text and therefore falls outside the main no-generation method.

This result is retained as evidence that direct answer readout is not a universal replacement for computation or reasoning.

For the complete methodology, additional metrics, candidate-order experiments, scoring comparisons, dataset hashes, and limitations, see:

- [Research Report](https://github.com/dbobo4/local-llm-probabilistic-decision-engine/blob/main/results/RESEARCH_REPORT.md)
- [Performance Benchmark Report](https://github.com/dbobo4/local-llm-probabilistic-decision-engine/blob/main/results/BENCHMARK_REPORT.md)

## Performance benchmark

A paired benchmark compared sequential and batch execution using:

```text
Model:     Qwen/Qwen2.5-1.5B-Instruct
GPU:       NVIDIA GeForce RTX 5070 Ti
Precision: torch.bfloat16
```

For the three-candidate benchmark:

| Metric | Sequential | Batch |
| --- | ---: | ---: |
| Median latency | 54.551 ms | 23.869 ms |
| P95 latency | 98.988 ms | 43.956 ms |

Median paired speedup:

```text
2.288x
```

Selected candidate agreement in that benchmark:

```text
100%
```

Maximum absolute probability difference:

```text
0.002875
```

Candidate-count scaling was also measured for 2, 3, 5, 10, and 20 candidates.

These results are specific to the tested hardware, model, prompt, candidate set, precision, software stack, and benchmark procedure. They are not universal performance guarantees.

See [results/BENCHMARK_REPORT.md](https://github.com/dbobo4/local-llm-probabilistic-decision-engine/blob/main/results/BENCHMARK_REPORT.md) for the complete benchmark.

## Candidate-order sensitivity

Candidate order is part of the model context.

In the routing calibration experiment, changing candidate order altered some decisions.

The smaller 1.5B model produced the same decision across all tested orders on:

```text
99 / 120 examples
```

The 3B model did so on:

```text
117 / 120 examples
```

The library therefore preserves the insertion order supplied by the application and does not silently sort candidates.

Applications should evaluate order sensitivity when it matters for their task.

## Model compatibility

The current backend uses:

```text
PyTorch
Hugging Face Transformers
Hugging Face Accelerate
Safetensors
```

The library is intended for compatible causal language models whose tokenizer provides the chat-template behavior required by the current prompt construction.

Model-family behavior can differ because of:

- tokenizer behavior
- chat templates
- vocabulary
- candidate tokenization
- numerical precision
- instruction tuning
- device support

Compatibility with one causal model does not imply identical behavior across all models.

See [MODEL_COMPATIBILITY.md](https://github.com/dbobo4/local-llm-probabilistic-decision-engine/blob/main/MODEL_COMPATIBILITY.md).

## Local and offline use

Model inference is performed by the locally loaded model.

However, passing a Hugging Face model ID without `local_files_only=True` may cause Transformers to contact the Hugging Face Hub to obtain missing files.

For explicitly offline operation, use either:

```text
a local model directory
```

or:

```python
local_files_only=True
```

with all required files already present locally.

Runnable example:

```text
examples/offline_model.py
```

## Examples

The repository includes:

```text
examples/basic_choice.py
examples/described_choice.py
examples/basic_boolean.py
examples/basic_rating.py
examples/text_rating.py
examples/decide.py
examples/offline_model.py
```

## Repository structure

```text
src/llm_decision_engine/   reusable Python package
examples/                  user-facing runnable examples
tests/                     automated tests
benchmarks/                benchmark and dataset tooling
experiments/               research experiments and diagnostics
results/                   committed benchmark and research results
```

## Development

Install development dependencies:

```text
python -m pip install -e ".[dev]"
```

Run tests:

```text
python -m pytest -q
```

Build the distribution:

```text
python -m build
```

Generate the performance report from committed benchmark results:

```text
python benchmarks/generate_report.py
```

Generate the research report:

```text
python benchmarks/generate_research_report.py
```

## Project scope

v0.1.0 focuses on a deliberately small public API:

```text
Choice
Boolean
Rating
decide()
```

The project does not currently attempt to provide every possible structured-output primitive or every local inference backend.

The API is pre-1.0 and may evolve in future releases.

## Independent implementation

This project is an independent implementation based on the author's own technical approach to the general problem.

Any inspiration from TypeSafe/Jev was limited to brief, publicly available promotional material. That material prompted the author to explore the general problem using their own technical approach. The architecture, methods, implementation, and source code in this repository were independently designed and developed.

The project was developed without access to TypeSafe/Jev internals, proprietary architecture details, training code, model weights, source code, or other non-public implementation information.

This repository does not reproduce or reverse-engineer a proprietary implementation and does not claim knowledge of one.

References to TypeSafe/Jev should not be interpreted as affiliation with, endorsement by, sponsorship by, or implementation of TypeSafe/Jev technology.

## Licensing

The code and original materials in this repository are licensed under the Apache License 2.0.

Third-party dependencies and model weights remain governed by their own licenses and terms.

Model weights are not redistributed with this package.

See:

- [LICENSE](https://github.com/dbobo4/local-llm-probabilistic-decision-engine/blob/main/LICENSE)
- [THIRD_PARTY_LICENSES.md](https://github.com/dbobo4/local-llm-probabilistic-decision-engine/blob/main/THIRD_PARTY_LICENSES.md)
- [MODEL_COMPATIBILITY.md](https://github.com/dbobo4/local-llm-probabilistic-decision-engine/blob/main/MODEL_COMPATIBILITY.md)

## Research and production use

The project is an experimental research-oriented library.

The committed benchmarks demonstrate specific behavior under specific controlled conditions. They should not be interpreted as guarantees of:

- universal accuracy
- universal calibration
- universal latency improvement
- equivalent behavior across models
- production suitability for every task

Applications should validate the method on representative data from their own domain before relying on its probabilities or decisions.
