# Model Compatibility

This project is designed to work with compatible open-weight causal language models. Model support and model licensing are separate concerns.

## Tested baseline

### Qwen/Qwen2.5-1.5B-Instruct

- Status: tested baseline
- Backend: Hugging Face Transformers / PyTorch
- Model license: Apache-2.0
- Model weights are downloaded from the original provider and are not redistributed with this package.

## Model responsibility

Users may select other compatible models through `DecisionEngine(model=...)`. Each model remains governed by its own license and terms. The fact that a model can technically be loaded by this library does not imply that this project grants permission to use, redistribute, modify, or commercialize that model.

## Compatibility goals

Planned compatibility work includes:

- additional Hugging Face causal language models
- arbitrary multi-token candidate scoring
- quantized local models
- later llama.cpp / GGUF support

Model-specific behavior, tokenization, chat templates, numerical precision, and licensing may differ between model families.
