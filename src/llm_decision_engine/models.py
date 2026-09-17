from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(slots=True)
class ModelBundle:
    tokenizer: object
    model: torch.nn.Module


def load_model(model_id: str) -> ModelBundle:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return ModelBundle(tokenizer=tokenizer, model=model)
