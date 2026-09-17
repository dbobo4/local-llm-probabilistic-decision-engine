# Local LLM Probabilistic Decision Engine

A local, open-source probabilistic decision engine built on top of open-weight LLMs. It replaces autoregressive output generation with direct candidate scoring, producing typed, normalized decision probabilities with zero generated output tokens.

## Independent implementation

This is a fully independent implementation based only on publicly described product behavior. I have no access to TypeSafe/Jev internals, architecture, training code, weights, or proprietary technical details. The design in this repository is my own approach to building a local probabilistic decision engine with similar externally observable behavior.

This project does not reproduce, reverse-engineer, or claim knowledge of Jev's proprietary architecture. Any architectural choices in this repository are independent design decisions.

## What it does

Instead of asking a language model to autoregressively generate a textual answer, the engine directly scores a constrained set of candidate decisions from the model's logits.

Example input:

```text
State: My account has not worked for three days and I am losing sales.

Question: Which department should handle this?

Candidates:
- technical support
- billing
- sales
```

Direct decision:

```text
technical support    92.84%
billing               6.73%
sales                 0.43%

Generated output tokens: 0
```

The current baseline performs a single model forward pass, extracts the relevant candidate logits, and normalizes them into a probability distribution over the allowed decision set.

## Basic usage

```python
from llm_decision_engine import DecisionEngine

engine = DecisionEngine(
    model="Qwen/Qwen2.5-1.5B-Instruct"
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
    "technical support": 0.9284,
    "billing": 0.0673,
    "sales": 0.0043,
}
```

## Current implementation status

The repository currently contains a validated single-token decision-scoring baseline and an installable Python package. The public API is already separated from the experimental implementation so that the scoring backend can evolve without changing normal library usage.

Current capabilities:

- Local open-weight Hugging Face models
- Direct candidate scoring from model logits
- Normalized candidate probability distributions
- Zero generated output tokens for the decision
- Structured `ChoiceResult` output
- Full-vocabulary candidate-mass measurement
- Unit-tested scoring mathematics
- Installable `llm_decision_engine` Python package

Current baseline limitation: candidate decisions are internally mapped to single-token labels (`A` through `Z`). This is temporary and will be replaced by arbitrary multi-token candidate scoring.

## Core idea

For a single-token candidate set with logits

```math
z = [z_1, z_2, ..., z_n]
```

the decision distribution is

```math
P(c_i | x, c \in C) = \frac{\exp(z_i)}{\sum_j \exp(z_j)}
```

where `x` is the model context and `C` is the allowed candidate set.

For arbitrary multi-token candidates, the planned scoring rule is based on sequence log-likelihood:

```math
S(c) = \sum_{t=1}^{T_c} \log P(c_t \mid x, c_1, \ldots, c_{t-1})
```

followed by normalization across candidate scores.

## Engine roadmap

```text
ENGINE
├── Choice
├── Boolean / proposition probability
├── Score
├── arbitrary multi-token candidates
├── batched candidate scoring
├── multiple questions per state
├── probability calibration
├── Hugging Face / PyTorch backend
└── later: llama.cpp / GGUF backend
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
python -m pip install -e .
```

Run the example:

```text
python examples/basic_choice.py
```

Run the tests:

```text
python -m pytest -q
```

## Model weights

Model weights are not distributed with this repository. Models are downloaded from their original providers and cached locally by the selected inference backend. Users are responsible for complying with the license and terms of the model they choose.

## Project status

Early development. The current implementation is a validated baseline; multi-token candidate scoring, calibration, benchmarking, and additional decision primitives are under active development.
