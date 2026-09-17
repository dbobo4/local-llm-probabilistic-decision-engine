from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ChoiceResult:
    probabilities: dict[str, float]
    scores: dict[str, float]
    selected: str
    generated_output_tokens: int
    scoring_method: str
    token_counts: dict[str, int]
