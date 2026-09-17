import torch

from .models import load_model
from .scoring import normalize_candidate_scores, score_causal_continuation
from .tokenization import batch_continuations, tokenize_continuation
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
        scoring: str = "sum",
        execution: str = "sequential",
    ) -> ChoiceResult:
        if len(candidates) < 2:
            raise ValueError("choice() requires at least two candidates.")

        if any(not candidate for candidate in candidates):
            raise ValueError("Candidates must not be empty.")

        if len(set(candidates)) != len(candidates):
            raise ValueError("Candidates must be unique.")

        if scoring not in {"sum", "mean"}:
            raise ValueError("scoring must be either 'sum' or 'mean'.")

        if execution not in {"sequential", "batch"}:
            raise ValueError("execution must be either 'sequential' or 'batch'.")

        candidate_list = "\n".join(
            f"- {candidate}" for candidate in candidates
        )

        user_prompt = f"""STATE:
{state}

QUESTION:
{question}

CANDIDATES:
{candidate_list}

Return exactly one candidate."""

        prefix = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

        tokenized_candidates = [
            tokenize_continuation(self.tokenizer, prefix, candidate)
            for candidate in candidates
        ]

        input_device = next(self.model.parameters()).device

        if execution == "sequential":
            scores = self._score_sequential(
                tokenized_candidates,
                scoring=scoring,
                device=input_device,
            )
        else:
            scores = self._score_batch(
                tokenized_candidates,
                scoring=scoring,
                device=input_device,
            )

        probabilities = normalize_candidate_scores(scores)

        probability_map = dict(zip(candidates, probabilities))
        score_map = dict(zip(candidates, scores))
        token_counts = {
            candidate: len(tokenized.target_token_ids)
            for candidate, tokenized in zip(candidates, tokenized_candidates)
        }

        selected_index = max(
            range(len(probabilities)),
            key=probabilities.__getitem__,
        )

        return ChoiceResult(
            probabilities=probability_map,
            scores=score_map,
            selected=candidates[selected_index],
            generated_output_tokens=0,
            scoring_method=scoring,
            token_counts=token_counts,
            execution_mode=execution,
        )

    def _score_sequential(self, tokenized_candidates, *, scoring, device):
        scores = []

        for tokenized in tokenized_candidates:
            input_ids = torch.tensor(
                [tokenized.input_ids],
                dtype=torch.long,
                device=device,
            )
            attention_mask = torch.ones_like(input_ids)

            with torch.inference_mode():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

            scores.append(
                score_causal_continuation(
                    outputs.logits[0],
                    tokenized.prefix_length,
                    tokenized.target_token_ids,
                    reduction=scoring,
                )
            )

        return scores

    def _score_batch(self, tokenized_candidates, *, scoring, device):
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id

        if pad_token_id is None:
            raise RuntimeError(
                "The tokenizer must define either pad_token_id or eos_token_id."
            )

        batch = batch_continuations(
            tokenized_candidates,
            pad_token_id=pad_token_id,
            device=device,
        )

        with torch.inference_mode():
            outputs = self.model(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
            )

        return [
            score_causal_continuation(
                outputs.logits[row],
                batch.prefix_lengths[row],
                batch.target_token_ids[row],
                reduction=scoring,
            )
            for row in range(len(tokenized_candidates))
        ]
