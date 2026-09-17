from llm_decision_engine import DecisionEngine


engine = DecisionEngine(
    model="Qwen/Qwen2.5-1.5B-Instruct",
)

result = engine.rating(
    state="The issue blocks the customer from using a core product feature.",
    question="How severe is this issue?",
    levels=[
        "minor",
        "moderate",
        "serious",
        "critical",
    ],
)

print("Probabilities:", result.probabilities)
print("Selected:", result.selected)
print("Expected value:", result.expected_value)
print("Generated output tokens:", result.generated_output_tokens)
