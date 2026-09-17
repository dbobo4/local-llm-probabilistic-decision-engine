import string

import torch

from .models import load_model
from .scoring import score_single_token_candidates
from .types import ChoiceResult


class DecisionEngine:
    def __init__(self, model: str):
        self.model_id = model
        bundle = load_model(model)
        self.tokenizer = bundle.tokenizer
        self.model = bundle.model

    def choice(
        self,
        *,
        state: str,
        question: str,
        candidates: list[str],
    ) -> ChoiceResult:
        if len(candidates) < 2:
            raise ValueError("choice() requires at least two candidates.")

        if len(candidates) > 26:
            raise ValueError(
                "The current single-token baseline supports at most 26 candidates."
            )

        labels = list(string.ascii_uppercase[: len(candidates)])
        options = "\n".join(
            f"{label} = {candidate}"
            for label, candidate in zip(labels, candidates)
        )

        if len(labels) == 2:
            label_instruction = f"{labels[0]} or {labels[1]}"
        else:
            label_instruction = ", ".join(labels[:-1]) + f", or {labels[-1]}"

        user_prompt = f"""STATE:
{state}

QUESTION:
{question}

OPTIONS:
{options}

Answer with exactly one label: {label_instruction}."""

        messages = [{"role": "user", "content": user_prompt}]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        input_device = next(self.model.parameters()).device
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        ).to(input_device)

        candidate_token_ids: list[int] = []

        for label in labels:
            token_ids = self.tokenizer.encode(
                label,
                add_special_tokens=False,
            )

            if len(token_ids) != 1:
                raise RuntimeError(
                    f"Internal label {label!r} is not a single token for this tokenizer."
                )

            candidate_token_ids.append(token_ids[0])

        with torch.inference_mode():
            outputs = self.model(**inputs)

        next_token_logits = outputs.logits[0, -1, :]

        logits, probabilities, candidate_mass = score_single_token_candidates(
            next_token_logits,
            candidate_token_ids,
        )

        probability_map = {
            candidate: probabilities[index].item()
            for index, candidate in enumerate(candidates)
        }

        logit_map = {
            candidate: logits[index].item()
            for index, candidate in enumerate(candidates)
        }

        selected_index = torch.argmax(probabilities).item()

        return ChoiceResult(
            probabilities=probability_map,
            logits=logit_map,
            selected=candidates[selected_index],
            generated_output_tokens=0,
            full_vocabulary_candidate_mass=candidate_mass,
        )
