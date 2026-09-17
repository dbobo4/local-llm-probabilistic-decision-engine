from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(slots=True)
class ModelBundle:
    tokenizer: object
    model: torch.nn.Module


def load_model(
    model_id: str,
    *,
    local_files_only: bool = False,
) -> ModelBundle:
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        local_files_only=local_files_only,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=local_files_only,
    )

    model.eval()

    return ModelBundle(
        tokenizer=tokenizer,
        model=model,
    )
