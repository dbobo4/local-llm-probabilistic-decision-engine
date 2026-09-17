import torch

from .models import load_model
from .scoring import normalize_candidate_scores, score_causal_continuation
from .tokenization import tokenize_continuation
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
    ) -> ChoiceResult:
        if len(candidates) < 2:
            raise ValueError("choice() requires at least two candidates.")

        if any(not candidate for candidate in candidates):
            raise ValueError("Candidates must not be empty.")

        if len(set(candidates)) != len(candidates):
            raise ValueError("Candidates must be unique.")

        if scoring not in {"sum", "mean"}:
            raise ValueError("scoring must be either 'sum' or 'mean'.")

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

        messages = [{"role": "user", "content": user_prompt}]

        prefix = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        input_device = next(self.model.parameters()).device

        scores: list[float] = []
        token_counts: dict[str, int] = {}

        for candidate in candidates:
            tokenized = tokenize_continuation(
                self.tokenizer,
                prefix,
                candidate,
            )

            input_ids = torch.tensor(
                [tokenized.input_ids],
                dtype=torch.long,
                device=input_device,
            )
            attention_mask = torch.ones_like(input_ids)

            with torch.inference_mode():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

            score = score_causal_continuation(
                outputs.logits[0],
                tokenized.prefix_length,
                tokenized.target_token_ids,
                reduction=scoring,
            )

            scores.append(score)
            token_counts[candidate] = len(tokenized.target_token_ids)

        probabilities = normalize_candidate_scores(scores)

        probability_map = dict(zip(candidates, probabilities))
        score_map = dict(zip(candidates, scores))

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
        )
