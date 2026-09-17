from llm_decision_engine import DecisionEngine


engine = DecisionEngine(model="Qwen/Qwen2.5-1.5B-Instruct")

result = engine.boolean(
    state="The payment was charged twice.",
    question="Should this be escalated?",
)

print(f"P(true):  {result.probability_true:.6f}")
print(f"P(false): {result.probability_false:.6f}")
print("Selected:", result.selected)
print("Scores:", result.scores)
print("Execution:", result.execution_mode)
print("Generated output tokens:", result.generated_output_tokens)
