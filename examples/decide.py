from llm_decision_engine import Boolean, Choice, DecisionEngine, Rating


engine = DecisionEngine(
    model="Qwen/Qwen2.5-1.5B-Instruct",
)

message = "I cannot log in and I need access immediately."

result = engine.decide(
    state=message,
    questions={
        "route": Choice(
            question="Which team should handle this request?",
            candidates={
                "billing": "Charges, invoices, payments, receipts, refunds, and unexpected fees.",
                "technical support": "Product failures, crashes, errors, broken features, or synchronization problems.",
                "account access": "Login, password, authentication, locked-account, or two-factor-access problems.",
            },
        ),
        "urgent": Boolean(
            question="Does this request require urgent attention?",
        ),
        "severity": Rating(
            question="How severe is the issue?",
            levels=[
                "minor",
                "moderate",
                "serious",
                "critical",
            ],
        ),
    },
)

route = result.results["route"]
urgent = result.results["urgent"]
severity = result.results["severity"]

print("Route:", route.selected)
print("Route probabilities:", route.probabilities)

print("\nUrgent:", urgent.selected)
print("P(True):", urgent.probability_true)

print("\nSeverity:", severity.selected)
print("Severity probabilities:", severity.probabilities)

print("\nTotal generated output tokens:", result.generated_output_tokens)
