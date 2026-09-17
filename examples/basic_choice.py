from llm_decision_engine import DecisionEngine


engine = DecisionEngine(
    model="Qwen/Qwen2.5-1.5B-Instruct",
)

result = engine.choice(
    state="The customer's account has not worked for three days and they are losing sales.",
    question="Which department should handle this?",
    candidates=[
        "technical support",
        "billing",
        "sales",
    ],
)

print("Probabilities:")
for candidate, probability in result.probabilities.items():
    print(f"  {candidate:<20} {probability:.6f}")

print(f"\nSelected: {result.selected}")
print(f"Generated output tokens: {result.generated_output_tokens}")
print(
    "Full-vocabulary candidate mass: "
    f"{result.full_vocabulary_candidate_mass:.6f}"
)
