import json
from pathlib import Path


RESULTS_DIR = Path("results")
OUTPUT_PATH = RESULTS_DIR / "RESEARCH_REPORT.md"


def load_json(name):
    return json.loads(
        (RESULTS_DIR / name).read_text(encoding="utf-8")
    )


def models_by_name(data):
    return {
        row["model"]: row
        for row in data["models"]
    }


def pct(value):
    return f"{value * 100:.1f}%"


routing_test = load_json("routing_v1_frozen_test.json")
definition_guided = load_json(
    "routing_v1_definition_guided_calibration.json"
)
candidate_order = load_json(
    "routing_v1_candidate_order_calibration.json"
)
sum_vs_mean = load_json(
    "routing_v1_sum_vs_mean_calibration.json"
)
binary_test = load_json("binary_quality_v2_test_final.json")

routing_models = models_by_name(routing_test)
guided_models = models_by_name(definition_guided)
order_models = models_by_name(candidate_order)
sum_mean_models = models_by_name(sum_vs_mean)
binary_models = models_by_name(binary_test)

model_names = [
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
]

lines = [
    "# Research Report",
    "",
    "This report summarizes committed research results for the Local LLM Probabilistic Decision Engine. It is generated from committed result JSON files and does not run model inference.",
    "",
    "## Research question",
    "",
    "> When can autoregressive generation be replaced by direct probabilistic decision inference?",
    "",
    "The main method scores complete candidate continuations directly under a causal language model. It does not autoregressively generate an answer.",
    "",
    "The experiments below are intended to identify both useful operating regions and failure boundaries. They do not establish universal equivalence between direct scoring and generation.",
    "",
    "## Main routing result",
    "",
    "The routing evaluation used a pre-specified, frozen protocol with six support categories and explicit category definitions. The test split was held out from model evaluation until the frozen run.",
    "",
    f"- Evaluation type: `{routing_test['evaluation_type']}`",
    f"- Dataset SHA256: `{routing_test['dataset_sha256']}`",
    "- Direct scoring: sum sequence log-likelihood, sequential execution",
    "- Direct generated output tokens: 0",
    "- Greedy control: deterministic generation with exact canonical-candidate parsing",
    "",
    "| Model | Direct accuracy | Top-2 | Greedy accuracy | Direct/greedy agreement | Direct output tokens | Greedy output tokens |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
]

for name in model_names:
    row = routing_models[name]
    direct = row["direct"]
    greedy = row["greedy_control"]

    lines.append(
        f"| `{name}` "
        f"| {pct(direct['accuracy'])} "
        f"| {pct(direct['top2_accuracy'])} "
        f"| {pct(greedy['accuracy'])} "
        f"| {pct(greedy['direct_agreement'])} "
        f"| {direct['generated_output_tokens']} "
        f"| {greedy['generated_output_tokens']} |"
    )

lines += [
    "",
    "On this controlled routing task, direct candidate scoring reproduced the greedy route decision almost perfectly for the 1.5B model and exactly for the 3B model, while the direct method produced no autoregressively generated output tokens.",
    "",
    "This result is specific to this benchmark, candidate schema, prompt protocol, and tested models. It is not a general proof that direct scoring and greedy generation are equivalent.",
    "",
    "This was not a blind benchmark: the benchmark generator and template source were known during benchmark construction. The test split was nevertheless frozen before model evaluation. It has now been evaluated and is therefore an opened test set; it should not be used for further tuning while still being described as untouched held-out evidence.",
    "",
    "## Effect of explicit candidate definitions",
    "",
    "Before the frozen test, calibration experiments compared bare candidate labels with the same labels accompanied by fixed semantic definitions.",
    "",
    "| Model | Bare accuracy | Definition-guided accuracy | Decisions changed | Errors fixed | Correct decisions broken |",
    "| --- | ---: | ---: | ---: | ---: | ---: |",
]

for name in model_names:
    row = guided_models[name]
    bare = row["bare"]["metrics"]
    guided = row["definition_guided"]["metrics"]
    switches = row["decision_switches"]

    lines.append(
        f"| `{name}` "
        f"| {pct(bare['accuracy'])} "
        f"| {pct(guided['accuracy'])} "
        f"| {switches['count']} "
        f"| {switches['fixed']} "
        f"| {switches['broken']} |"
    )

lines += [
    "",
    "The calibration result supports an important design principle for the public API: candidate descriptions can provide an explicit decision schema while the candidate key itself remains the scored output.",
    "",
    "This was a calibration-stage result on a synthetic routing benchmark, not an independent real-world routing evaluation.",
    "",
    "## Candidate-order sensitivity",
    "",
    "Candidate order is part of the prompt seen by the model. Reordering the same six candidates changed some decisions, especially for the smaller model.",
    "",
    "| Model | Canonical accuracy | Reversed | Rotated | Fixed shuffle | Stable across all tested orders |",
    "| --- | ---: | ---: | ---: | ---: | ---: |",
]

for name in model_names:
    row = order_models[name]
    stable_count = row["stable_across_all_orders"]["count"]
    stable_rate = stable_count / 120

    lines.append(
        f"| `{name}` "
        f"| {pct(row['canonical']['accuracy'])} "
        f"| {pct(row['variants']['reversed']['accuracy'])} "
        f"| {pct(row['variants']['rotated_2']['accuracy'])} "
        f"| {pct(row['variants']['fixed_shuffle']['accuracy'])} "
        f"| {stable_count}/120 ({pct(stable_rate)}) |"
    )

lines += [
    "",
    "The library therefore preserves user-supplied insertion order rather than silently sorting candidates. Applications that depend on stable behavior should keep candidate ordering deterministic and evaluate order sensitivity on their own task.",
    "",
    "## Sum versus mean scoring",
    "",
    "The default method uses the full candidate sequence log-likelihood:",
    "",
    "```math",
    r"S(c) = \sum_t \log P(c_t \mid x, c_{<t})",
    "```",
    "",
    "A length-normalized mean-log-probability alternative was also tested.",
    "",
    "| Model | Sum accuracy | Mean accuracy | Decision switches |",
    "| --- | ---: | ---: | ---: |",
]

for name in model_names:
    row = sum_mean_models[name]

    lines.append(
        f"| `{name}` "
        f"| {pct(row['sum']['metrics']['accuracy'])} "
        f"| {pct(row['mean']['metrics']['accuracy'])} "
        f"| {row['decision_switches']['count']} |"
    )

lines += [
    "",
    "Mean scoring did not materially change the smaller model's decisions and changed only one decision for the 3B model in this calibration experiment. The result does not justify replacing sum scoring as the default.",
    "",
    "Sum remains the principled default because it corresponds to the probability of the complete candidate token sequence in log-space. Mean scoring is exposed as an alternative diagnostic.",
    "",
    "## Boundary condition: binary arithmetic and verification tasks",
    "",
    "A separate pre-specified 120-example test, held out from model evaluation until the frozen run, tested the dedicated direct True/False protocol on arithmetic and verification-style tasks.",
    "",
    f"- Evaluation type: `{binary_test['evaluation_type']}`",
    f"- Protocol frozen before test: `{str(binary_test['protocol_frozen_before_test']).lower()}`",
    f"- Dataset SHA256: `{binary_test['dataset_sha256']}`",
    f"- Main method: {binary_test['main_method']}",
    f"- Main-method generated output tokens: {binary_test['main_method_generated_output_tokens']}",
    f"- Diagnostic control: {binary_test['diagnostic_control']}",
    "",
    "| Model | Direct accuracy | Direct Brier | Direct NLL | Reasoning-conditioned accuracy |",
    "| --- | ---: | ---: | ---: | ---: |",
]

for name in model_names:
    row = binary_models[name]
    direct = row["direct"]["evaluation"]
    reasoned = row["reasoning_conditioned"]["evaluation"]

    lines.append(
        f"| `{name}` "
        f"| {pct(direct['accuracy'])} "
        f"| {direct['mean_brier']:.3f} "
        f"| {direct['mean_nll']:.3f} "
        f"| {pct(reasoned['accuracy'])} |"
    )

lines += [
    "",
    "This benchmark is important negative evidence. Direct answer readout can fail when the task requires intermediate computation that is not already represented strongly enough at the answer position.",
    "",
    "The reasoning-conditioned control generated intermediate verification text before performing the final direct readout. Its improvement shows that additional computation can matter, but that control is outside the main no-generation method.",
    "",
    "The project therefore does not claim that direct scoring should replace generation on reasoning-heavy tasks.",
    "",
    "## What the evidence currently supports",
    "",
    "- On the tested routing task, direct scoring worked well for constrained semantic decisions where the allowed outputs and decision schema were known in advance.",
    "- Explicit candidate definitions can substantially improve decisions when short labels are semantically ambiguous.",
    "- Multi-token candidates can be scored directly as complete causal continuations.",
    "- Direct candidate scoring can closely reproduce greedy output decisions on the tested routing task without generating answer tokens.",
    "- Candidate ordering can influence results and should be treated as part of the inference protocol.",
    "- Direct answer readout can be substantially weaker when intermediate computation is required.",
    "",
    "## What the evidence does not establish",
    "",
    "- It does not prove universal equivalence between direct scoring and autoregressive generation.",
    "- It does not show that direct scoring is always faster or uses less total compute.",
    "- It does not establish that normalized candidate probabilities are calibrated real-world probabilities of correctness.",
    "- It does not establish production routing quality outside the synthetic benchmark.",
    "- It does not establish compatibility or equivalent behavior across all causal language models.",
    "",
    "## Probability interpretation",
    "",
    "The engine normalizes scores across the supplied candidate set. The resulting values are restricted-choice model probabilities over those candidates.",
    "",
    "A high normalized probability should not automatically be interpreted as a calibrated probability that the selected candidate is objectively correct. Calibration must be measured separately on representative labeled data.",
    "",
    "## Performance benchmarks",
    "",
    "Sequential-versus-batch latency and candidate-count scaling are reported separately in `results/BENCHMARK_REPORT.md`.",
    "",
    "Those performance measurements are hardware-, model-, precision-, prompt-, and candidate-set-specific and are not universal speed guarantees.",
    "",
    "## Reproduction",
    "",
    "The repository retains the benchmark generators, frozen datasets, evaluation scripts, result JSON files, dataset hashes, and relevant protocol metadata needed to inspect and reproduce the experiments.",
    "",
    "Key result files:",
    "",
    "- `results/routing_v1_frozen_test.json`",
    "- `results/routing_v1_definition_guided_calibration.json`",
    "- `results/routing_v1_candidate_order_calibration.json`",
    "- `results/routing_v1_sum_vs_mean_calibration.json`",
    "- `results/binary_quality_v2_test_final.json`",
    "- `results/BENCHMARK_REPORT.md`",
    "",
]

OUTPUT_PATH.write_text(
    "\n".join(lines),
    encoding="utf-8",
)

print("Generated:", OUTPUT_PATH)
