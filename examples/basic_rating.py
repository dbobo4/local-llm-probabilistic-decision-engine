from llm_decision_engine import DecisionEngine


engine = DecisionEngine(model="Qwen/Qwen2.5-1.5B-Instruct")

result = engine.rating(
    state="The response is mostly correct but contains one minor factual error.",
    question="Rate the reliability from 1 to 5, where 1 is very unreliable and 5 is very reliable.",
    levels=[1, 2, 3, 4, 5],
)

print("Probabilities:", result.probabilities)
print("Selected:", result.selected)
print(f"Expected value: {result.expected_value:.6f}")
print("Scores:", result.scores)
print("Execution:", result.execution_mode)
print("Generated output tokens:", result.generated_output_tokens)
