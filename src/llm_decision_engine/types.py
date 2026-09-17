from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ChoiceResult:
    probabilities: dict[str, float]
    logits: dict[str, float]
    selected: str
    generated_output_tokens: int
    full_vocabulary_candidate_mass: float
