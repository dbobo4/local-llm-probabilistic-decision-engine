import pytest

from llm_decision_engine.tokenization import (
    TokenizedContinuation,
    batch_continuations,
    tokenize_continuation,
)


class StableTokenizer:
    def encode(self, text, add_special_tokens=False):
        mapping = {
            "Decision:": [10, 11],
            "Decision: technical support": [10, 11, 21, 22],
        }
        return mapping[text]


class BoundaryChangingTokenizer:
    def encode(self, text, add_special_tokens=False):
        mapping = {
            "Decision:": [10, 11],
            "Decision:technical": [10, 99],
        }
        return mapping[text]


def test_tokenize_continuation_extracts_only_candidate_tokens():
    result = tokenize_continuation(
        StableTokenizer(),
        "Decision:",
        " technical support",
    )

    assert result.input_ids == [10, 11, 21, 22]
    assert result.prefix_length == 2
    assert result.target_token_ids == [21, 22]


def test_tokenize_continuation_rejects_boundary_changes():
    with pytest.raises(ValueError, match="boundary changed"):
        tokenize_continuation(
            BoundaryChangingTokenizer(),
            "Decision:",
            "technical",
        )


def test_batch_continuations_right_pads_sequences():
    continuations = [
        TokenizedContinuation(
            input_ids=[10, 11, 21, 22],
            prefix_length=2,
            target_token_ids=[21, 22],
        ),
        TokenizedContinuation(
            input_ids=[10, 11, 31],
            prefix_length=2,
            target_token_ids=[31],
        ),
    ]

    batch = batch_continuations(
        continuations,
        pad_token_id=0,
    )

    assert batch.input_ids.tolist() == [
        [10, 11, 21, 22],
        [10, 11, 31, 0],
    ]
    assert batch.attention_mask.tolist() == [
        [1, 1, 1, 1],
        [1, 1, 1, 0],
    ]
    assert batch.prefix_lengths == [2, 2]
    assert batch.target_token_ids == [[21, 22], [31]]


def test_batch_continuations_requires_input():
    with pytest.raises(ValueError, match="must not be empty"):
        batch_continuations([], pad_token_id=0)


def test_tokenize_continuation_rejects_prefix_with_no_tokens():
    class EmptyPrefixTokenizer:
        def encode(self, text, add_special_tokens=False):
            mapping = {
                "Prompt": [],
                "Prompt candidate": [42],
            }
            return mapping[text]

    with pytest.raises(ValueError, match="prefix produced no tokens"):
        tokenize_continuation(
            EmptyPrefixTokenizer(),
            "Prompt",
            " candidate",
        )
