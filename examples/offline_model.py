from llm_decision_engine import DecisionEngine


# Option 1: load from a local model directory.
engine = DecisionEngine(
    model=r"D:\models\Qwen2.5-1.5B-Instruct",
    local_files_only=True,
)

# Option 2: use a Hugging Face model ID but forbid network access.
# This requires the model files to already exist in the local Hugging Face cache.
#
# engine = DecisionEngine(
#     model="Qwen/Qwen2.5-1.5B-Instruct",
#     local_files_only=True,
# )

result = engine.choice(
    state="The customer wants a refund for a duplicate charge.",
    question="Which team should handle this?",
    candidates=[
        "billing",
        "technical support",
        "sales",
    ],
)

print(result.probabilities)
print("Selected:", result.selected)
print("Generated output tokens:", result.generated_output_tokens)
