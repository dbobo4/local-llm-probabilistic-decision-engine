from llm_decision_engine import DecisionEngine


engine = DecisionEngine(
    model="Qwen/Qwen2.5-1.5B-Instruct",
)

result = engine.choice(
    state="The customer account has not worked for three days and they are losing sales.",
    question="Which department should handle this?",
    candidates=[
        "technical support",
        "billing",
        "sales",
    ],
)

print("Probabilities:")
for candidate, probability in result.probabilities.items():
    print(
        f"  {candidate:<20} "
        f"{probability:.6f} "
        f"score={result.scores[candidate]:.6f} "
        f"tokens={result.token_counts[candidate]}"
    )

print(f"\nSelected: {result.selected}")
print(f"Scoring method: {result.scoring_method}")
print(f"Generated output tokens: {result.generated_output_tokens}")
