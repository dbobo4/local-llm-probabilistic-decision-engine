import pytest

from llm_decision_engine.tokenization import tokenize_continuation


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
