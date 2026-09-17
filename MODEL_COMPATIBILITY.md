# Model Compatibility

This project is designed for compatible causal language models that can be loaded through Hugging Face Transformers and used with the inference protocol implemented by this library.

Technical compatibility and model licensing are separate concerns.

## Tested models

The following models have been used in the repository's current v0.1.0 experiments and benchmarks:

### Qwen/Qwen2.5-1.5B-Instruct

- Status: tested
- Backend: Hugging Face Transformers / PyTorch
- Used in routing, binary-decision, scoring, and performance experiments.

### Qwen/Qwen2.5-3B-Instruct

- Status: tested
- Backend: Hugging Face Transformers / PyTorch
- Used in routing, binary-decision, and scoring experiments.

Testing these models does not imply that every model in the same family, or every causal language model supported by Transformers, will behave identically.

## Current model requirements

The current implementation expects a causal language model and tokenizer that are compatible with:

- `AutoTokenizer.from_pretrained(...)`
- `AutoModelForCausalLM.from_pretrained(...)`
- the tokenizer chat-template behavior used by the current prompt construction
- direct access to next-token logits
- causal continuation scoring

Candidate answers may contain multiple tokens.

The library scores the complete candidate continuation token by token using causal sequence likelihood. Multi-token candidate scoring is therefore part of the current implementation and is not a planned-only feature.

## Model loading

`DecisionEngine` requires a model to be supplied explicitly.

A Hugging Face model identifier can be used:

```python
engine = DecisionEngine(
    model="Qwen/Qwen2.5-1.5B-Instruct",
)
```

A local model directory can also be used:

```python
engine = DecisionEngine(
    model="./models/Qwen2.5-1.5B-Instruct",
)
```

For offline-only loading:

```python
engine = DecisionEngine(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    local_files_only=True,
)
```

With `local_files_only=True`, all required model and tokenizer files must already exist locally, either in the Hugging Face cache or in the supplied local directory.

Model weights are not bundled with this package.

## Current loading behavior

The current loader uses:

```text
torch.bfloat16
device_map="auto"
```

Actual hardware compatibility therefore depends on the installed PyTorch version, device support, model architecture, and available memory.

The repository's successful experiments should not be interpreted as a guarantee that every supported Python environment can run every model on every hardware configuration.

## Compatibility differences between models

Different model families may behave differently because of:

- tokenizer behavior
- chat-template format
- vocabulary and token boundaries
- candidate tokenization
- instruction tuning
- numerical precision
- device backend
- model architecture
- model size

Candidate probabilities and selected decisions can therefore differ across model families even when the same application-level prompt and candidates are used.

Applications should validate each intended model on representative data.

## Current scope

The v0.1.0 backend currently targets Hugging Face Transformers causal language models.

The following are not currently claimed as supported backends:

- llama.cpp
- GGUF-native inference
- arbitrary quantized inference backends

Support for those backends may be investigated in future versions, but they are outside the current v0.1.0 compatibility claim.

## Model licensing and responsibility

Each model remains governed by its own license, terms, and usage restrictions.

Technical compatibility with this library does not grant permission to use, redistribute, modify, or commercialize a model.

Users are responsible for reviewing the terms associated with the specific model they choose.

This package does not redistribute model weights.
