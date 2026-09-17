from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ChoiceResult:
    probabilities: dict[str, float]
    scores: dict[str, float]
    selected: str
    generated_output_tokens: int
    scoring_method: str
    token_counts: dict[str, int]
    execution_mode: str

@dataclass(frozen=True, slots=True)
class BooleanResult:
    probability_true: float
    probability_false: float
    selected: bool
    scores: dict[str, float]
    generated_output_tokens: int
    scoring_method: str
    execution_mode: str

@dataclass(frozen=True, slots=True)
class RatingResult:
    probabilities: dict[int, float]
    expected_value: float
    selected: int
    scores: dict[int, float]
    generated_output_tokens: int
    scoring_method: str
    execution_mode: str
