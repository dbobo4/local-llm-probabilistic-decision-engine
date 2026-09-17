from types import SimpleNamespace

import pytest
import torch

from llm_decision_engine import DecisionEngine


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "P"

    def encode(self, text, add_special_tokens=False):
        mapping = {
            "P": [10, 11],
            "Palpha": [10, 11, 1, 2],
            "Pbeta": [10, 11, 3],
            "Pyes": [10, 11, 1],
            "PTrue": [10, 11, 1],
            "Pno": [10, 11, 3],
            "PFalse": [10, 11, 3],
            "P1": [10, 11, 1],
            "P2": [10, 11, 2],
            "P3": [10, 11, 3],
        }
        return mapping[text]


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.calls = 0

    def forward(self, input_ids, attention_mask):
        self.calls += 1
        batch_size, sequence_length = input_ids.shape
        logits = torch.zeros(
            (batch_size, sequence_length, 4),
            dtype=torch.float32,
            device=input_ids.device,
        )

        for row in range(batch_size):
            if input_ids[row, 2].item() == 1:
                logits[row, 1, 1] = 4.0
                logits[row, 2, 2] = 4.0
            elif input_ids[row, 2].item() == 3:
                logits[row, 1, 3] = 1.0

        return SimpleNamespace(logits=logits)


def make_engine():
    engine = DecisionEngine.__new__(DecisionEngine)
    engine.model_id = "fake"
    engine.tokenizer = FakeTokenizer()
    engine.model = FakeModel()
    return engine


def test_choice_requires_at_least_two_candidates():
    engine = DecisionEngine.__new__(DecisionEngine)

    with pytest.raises(ValueError, match="at least two candidates"):
        engine.choice(
            state="test", question="test", candidates=["one"]
        )


def test_choice_rejects_duplicate_candidates():
    engine = DecisionEngine.__new__(DecisionEngine)

    with pytest.raises(ValueError, match="unique"):
        engine.choice(
            state="test", question="test", candidates=["x", "x"]
        )


def test_choice_rejects_invalid_scoring_method():
    engine = DecisionEngine.__new__(DecisionEngine)

    with pytest.raises(ValueError, match="scoring"):
        engine.choice(
            state="test",
            question="test",
            candidates=["x", "y"],
            scoring="invalid",
        )


def test_choice_rejects_invalid_execution_mode():
    engine = DecisionEngine.__new__(DecisionEngine)

    with pytest.raises(ValueError, match="execution"):
        engine.choice(
            state="test",
            question="test",
            candidates=["x", "y"],
            execution="invalid",
        )


def test_sequential_execution_uses_one_forward_per_candidate():
    engine = make_engine()

    result = engine.choice(
        state="state",
        question="question",
        candidates=["alpha", "beta"],
        execution="sequential",
    )

    assert engine.model.calls == 2
    assert result.execution_mode == "sequential"
    assert result.selected == "alpha"


def test_batch_execution_uses_one_model_forward():
    engine = make_engine()

    result = engine.choice(
        state="state",
        question="question",
        candidates=["alpha", "beta"],
        execution="batch",
    )

    assert engine.model.calls == 1
    assert result.execution_mode == "batch"
    assert result.selected == "alpha"


def test_sequential_and_batch_match_in_float32_reference():
    sequential_engine = make_engine()
    batch_engine = make_engine()

    sequential = sequential_engine.choice(
        state="state",
        question="question",
        candidates=["alpha", "beta"],
        execution="sequential",
    )
    batched = batch_engine.choice(
        state="state",
        question="question",
        candidates=["alpha", "beta"],
        execution="batch",
    )

    assert sequential.scores == pytest.approx(batched.scores, abs=1e-6)
    assert sequential.probabilities == pytest.approx(
        batched.probabilities,
        abs=1e-6,
    )
    assert sequential.token_counts == batched.token_counts
    assert sequential.selected == batched.selected

def test_boolean_returns_typed_probabilities():
    engine = make_engine()

    result = engine.boolean(
        state="state",
        question="question",
    )

    assert result.probability_true > result.probability_false
    assert result.selected is True
    assert result.generated_output_tokens == 0
    assert result.scoring_method == "sum"
    assert result.execution_mode == "sequential"
    assert engine.model.calls == 2


def test_boolean_supports_batch_execution():
    engine = make_engine()

    result = engine.boolean(
        state="state",
        question="question",
        execution="batch",
    )

    assert result.selected is True
    assert result.execution_mode == "batch"
    assert engine.model.calls == 1

def test_rating_returns_distribution_and_expected_value():
    engine = make_engine()

    result = engine.rating(
        state="state",
        question="question",
        scale=[1, 2, 3],
    )

    assert set(result.probabilities) == {1, 2, 3}
    assert sum(result.probabilities.values()) == pytest.approx(1.0)
    assert result.selected == 1
    assert result.expected_value == pytest.approx(
        sum(
            value * probability
            for value, probability in result.probabilities.items()
        )
    )
    assert result.generated_output_tokens == 0
    assert result.execution_mode == "sequential"
    assert engine.model.calls == 3


def test_rating_supports_batch_execution():
    engine = make_engine()

    result = engine.rating(
        state="state",
        question="question",
        scale=[1, 2, 3],
        execution="batch",
    )

    assert result.selected == 1
    assert result.execution_mode == "batch"
    assert engine.model.calls == 1


def test_rating_rejects_duplicate_scale_values():
    engine = make_engine()

    with pytest.raises(ValueError, match="unique"):
        engine.rating(
            state="state",
            question="question",
            scale=[1, 1],
        )


def test_rating_rejects_non_integer_scale_values():
    engine = make_engine()

    with pytest.raises(TypeError, match="integers"):
        engine.rating(
            state="state",
            question="question",
            scale=[1, 2.5],
        )

def test_rating_requires_at_least_two_scale_values():
    engine = make_engine()

    with pytest.raises(ValueError, match="at least two"):
        engine.rating(
            state="state",
            question="question",
            scale=[1],
        )


def test_rating_rejects_boolean_scale_values():
    engine = make_engine()

    with pytest.raises(TypeError, match="integers"):
        engine.rating(
            state="state",
            question="question",
            scale=[1, True],
        )

def test_boolean_rejects_invalid_scoring_method():
    engine = make_engine()

    with pytest.raises(ValueError, match="scoring"):
        engine.boolean(
            state="state",
            question="question",
            scoring="invalid",
        )


def test_boolean_rejects_invalid_execution_mode():
    engine = make_engine()

    with pytest.raises(ValueError, match="execution"):
        engine.boolean(
            state="state",
            question="question",
            execution="invalid",
        )


def test_boolean_preserves_semantic_score_keys():
    engine = make_engine()

    result = engine.boolean(
        state="state",
        question="question",
    )

    assert set(result.scores) == {"yes", "no"}

def test_boolean_uses_true_false_verbalizers():
    engine = make_engine()
    seen = []
    original_encode = engine.tokenizer.encode

    def recording_encode(text, add_special_tokens=False):
        seen.append(text)
        return original_encode(text, add_special_tokens=add_special_tokens)

    engine.tokenizer.encode = recording_encode

    engine.boolean(
        state="state",
        question="question",
    )

    assert "PTrue" in seen
    assert "PFalse" in seen
    assert "Pyes" not in seen
    assert "Pno" not in seen

def test_choice_list_prompt_remains_unchanged():
    engine = make_engine()
    prompts = []
    original_apply_chat_template = engine.tokenizer.apply_chat_template

    def recording_apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    ):
        prompts.append(messages[0]["content"])
        return original_apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
        )

    engine.tokenizer.apply_chat_template = recording_apply_chat_template

    engine.choice(
        state="state",
        question="question",
        candidates=["alpha", "beta"],
    )

    assert prompts == [
        """STATE:
state

QUESTION:
question

CANDIDATES:
- alpha
- beta

Return exactly one candidate."""
    ]


def test_choice_supports_candidate_descriptions():
    engine = make_engine()
    prompts = []
    original_apply_chat_template = engine.tokenizer.apply_chat_template

    def recording_apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    ):
        prompts.append(messages[0]["content"])
        return original_apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
        )

    engine.tokenizer.apply_chat_template = recording_apply_chat_template

    result = engine.choice(
        state="state",
        question="question",
        candidates={
            "alpha": "The first semantic option.",
            "beta": "The second semantic option.",
        },
    )

    assert result.selected == "alpha"
    assert list(result.probabilities) == ["alpha", "beta"]
    assert list(result.scores) == ["alpha", "beta"]
    assert list(result.token_counts) == ["alpha", "beta"]

    assert prompts == [
        """STATE:
state

QUESTION:
question

CANDIDATE DEFINITIONS:
- alpha: The first semantic option.
- beta: The second semantic option.

CANDIDATES:
- alpha
- beta

Return exactly one candidate."""
    ]


def test_choice_candidate_descriptions_preserve_insertion_order():
    engine = make_engine()
    prompts = []
    original_apply_chat_template = engine.tokenizer.apply_chat_template

    def recording_apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    ):
        prompts.append(messages[0]["content"])
        return original_apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
        )

    engine.tokenizer.apply_chat_template = recording_apply_chat_template

    result = engine.choice(
        state="state",
        question="question",
        candidates={
            "beta": "Second.",
            "alpha": "First.",
        },
    )

    assert list(result.probabilities) == ["beta", "alpha"]

    prompt = prompts[0]

    assert prompt.index("- beta: Second.") < prompt.index("- alpha: First.")
    assert prompt.index(
        "CANDIDATES:\n- beta\n- alpha"
    ) != -1


def test_choice_rejects_empty_candidate_description():
    engine = make_engine()

    with pytest.raises(
        ValueError,
        match="descriptions must be non-empty strings",
    ):
        engine.choice(
            state="state",
            question="question",
            candidates={
                "alpha": "Valid description.",
                "beta": "   ",
            },
        )

