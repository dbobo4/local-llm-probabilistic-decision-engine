import torch

from .models import load_model
from .scoring import normalize_candidate_scores, score_causal_continuation
from .tokenization import batch_continuations, tokenize_continuation
from .types import (
    Boolean,
    BooleanResult,
    Choice,
    ChoiceResult,
    DecisionResult,
    Rating,
    RatingResult,
)


class DecisionEngine:
    def __init__(
        self,
        model: str,
        *,
        local_files_only: bool = False,
    ):
        self.model_id = model
        self.local_files_only = local_files_only

        bundle = load_model(
            model,
            local_files_only=local_files_only,
        )

        self.tokenizer = bundle.tokenizer
        self.model = bundle.model

    def choice(
        self,
        *,
        state: str,
        question: str,
        candidates: list[str] | dict[str, str],
        scoring: str = "sum",
        execution: str = "sequential",
    ) -> ChoiceResult:
        if isinstance(candidates, dict):
            candidate_names = list(candidates)
            candidate_definitions = candidates
        else:
            candidate_names = candidates
            candidate_definitions = None

        if len(candidate_names) < 2:
            raise ValueError("choice() requires at least two candidates.")

        if any(not candidate for candidate in candidate_names):
            raise ValueError("Candidates must not be empty.")

        if len(set(candidate_names)) != len(candidate_names):
            raise ValueError("Candidates must be unique.")

        if candidate_definitions is not None and any(
            not isinstance(description, str) or not description.strip()
            for description in candidate_definitions.values()
        ):
            raise ValueError(
                "Candidate descriptions must be non-empty strings."
            )

        if scoring not in {"sum", "mean"}:
            raise ValueError("scoring must be either 'sum' or 'mean'.")

        if execution not in {"sequential", "batch"}:
            raise ValueError("execution must be either 'sequential' or 'batch'.")

        candidate_list = "\n".join(
            f"- {candidate}" for candidate in candidate_names
        )

        if candidate_definitions is None:
            user_prompt = f"""STATE:
{state}

QUESTION:
{question}

CANDIDATES:
{candidate_list}

Return exactly one candidate."""
        else:
            definition_list = "\n".join(
                f"- {candidate}: {description}"
                for candidate, description in candidate_definitions.items()
            )

            user_prompt = f"""STATE:
{state}

QUESTION:
{question}

CANDIDATE DEFINITIONS:
{definition_list}

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
            for candidate in candidate_names
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

        probability_map = dict(zip(candidate_names, probabilities))
        score_map = dict(zip(candidate_names, scores))
        token_counts = {
            candidate: len(tokenized.target_token_ids)
            for candidate, tokenized in zip(
                candidate_names,
                tokenized_candidates,
            )
        }

        selected_index = max(
            range(len(probabilities)),
            key=probabilities.__getitem__,
        )

        return ChoiceResult(
            probabilities=probability_map,
            scores=score_map,
            selected=candidate_names[selected_index],
            generated_output_tokens=0,
            scoring_method=scoring,
            token_counts=token_counts,
            execution_mode=execution,
        )

    def boolean(
        self,
        *,
        state: str,
        question: str,
        scoring: str = "sum",
        execution: str = "sequential",
    ) -> BooleanResult:
        if scoring not in {"sum", "mean"}:
            raise ValueError("scoring must be either 'sum' or 'mean'.")

        if execution not in {"sequential", "batch"}:
            raise ValueError("execution must be either 'sequential' or 'batch'.")

        user_prompt = f"""STATE:
{state}

QUESTION:
{question}

Answer exactly one word: True or False."""

        prefix = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

        candidates = ["True", "False"]

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

        return BooleanResult(
            probability_true=probabilities[0],
            probability_false=probabilities[1],
            selected=probabilities[0] >= probabilities[1],
            scores={
                "yes": scores[0],
                "no": scores[1],
            },
            generated_output_tokens=0,
            scoring_method=scoring,
            execution_mode=execution,
        )
    def rating(
        self,
        *,
        state: str,
        question: str,
        levels: list[str] | list[int],
        scoring: str = "sum",
        execution: str = "sequential",
    ) -> RatingResult:
        if len(levels) < 2:
            raise ValueError("rating() requires at least two levels.")

        if any(
            isinstance(level, bool)
            or not isinstance(level, (str, int))
            for level in levels
        ):
            raise TypeError(
                "Rating levels must be strings or integers."
            )

        level_types = {type(level) for level in levels}
        if len(level_types) != 1:
            raise TypeError(
                "Rating levels must all use the same type."
            )

        if any(
            isinstance(level, str) and not level.strip()
            for level in levels
        ):
            raise ValueError("Rating levels must not be empty.")

        if len(set(levels)) != len(levels):
            raise ValueError("Rating levels must be unique.")

        candidates = [str(level) for level in levels]

        choice_result = self.choice(
            state=state,
            question=question,
            candidates=candidates,
            scoring=scoring,
            execution=execution,
        )

        probabilities = {
            level: choice_result.probabilities[str(level)]
            for level in levels
        }

        scores = {
            level: choice_result.scores[str(level)]
            for level in levels
        }

        selected = next(
            level
            for level in levels
            if str(level) == choice_result.selected
        )

        expected_value = None

        if all(isinstance(level, int) for level in levels):
            expected_value = sum(
                level * probabilities[level]
                for level in levels
            )

        return RatingResult(
            probabilities=probabilities,
            expected_value=expected_value,
            selected=selected,
            scores=scores,
            generated_output_tokens=choice_result.generated_output_tokens,
            scoring_method=choice_result.scoring_method,
            execution_mode=choice_result.execution_mode,
        )

    def decide(
        self,
        *,
        state: str,
        questions: dict[str, Choice | Boolean | Rating],
    ) -> DecisionResult:
        if not questions:
            raise ValueError("decide() requires at least one question.")

        if any(
            not isinstance(name, str) or not name.strip()
            for name in questions
        ):
            raise ValueError(
                "Decision question names must be non-empty strings."
            )

        results = {}

        for name, specification in questions.items():
            if isinstance(specification, Choice):
                result = self.choice(
                    state=state,
                    question=specification.question,
                    candidates=specification.candidates,
                    scoring=specification.scoring,
                    execution=specification.execution,
                )
            elif isinstance(specification, Boolean):
                result = self.boolean(
                    state=state,
                    question=specification.question,
                    scoring=specification.scoring,
                    execution=specification.execution,
                )
            elif isinstance(specification, Rating):
                result = self.rating(
                    state=state,
                    question=specification.question,
                    levels=specification.levels,
                    scoring=specification.scoring,
                    execution=specification.execution,
                )
            else:
                raise TypeError(
                    "Each decide() question must be a "
                    "Choice, Boolean, or Rating specification."
                )

            results[name] = result

        return DecisionResult(
            results=results,
            generated_output_tokens=sum(
                result.generated_output_tokens
                for result in results.values()
            ),
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
