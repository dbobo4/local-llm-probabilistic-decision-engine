from llm_decision_engine import DecisionEngine


engine = DecisionEngine(
    model="Qwen/Qwen2.5-1.5B-Instruct",
)

result = engine.choice(
    state="The customer cannot sign in after resetting their password twice.",
    question="Which support category best matches this request?",
    candidates={
        "billing": "Charges, invoices, payments, receipts, refunds, and unexpected fees.",
        "technical support": "Product failures, crashes, errors, broken features, or synchronization problems.",
        "account access": "Login, password, authentication, locked-account, or two-factor-access problems.",
    },
)

print("Probabilities:")
for candidate, probability in result.probabilities.items():
    print(f"  {candidate:<20} {probability:.6f}")

print("\nSelected:", result.selected)
print("Generated output tokens:", result.generated_output_tokens)
