# MULTI-Bench: A Multi-Turn Interactive Benchmark for Assessing Emotional Intelligence ability of Spoken Dialogue Models

- Year: 2025
- Venue: arXiv (submitted to ICASSP 2026)
- Authors: Guoqiang Hu et al. (see arXiv for full author list)
- URL: https://arxiv.org/abs/2511.00850
- BibTeX key (if we add it): multi_bench_hu_2025
- Tags: multi-turn, spoken-dialogue, emotional-intelligence, benchmark, evaluation, interactive

## One-sentence takeaway

Multi-Bench proposes a two-tier (basic/advanced) benchmark with an interactive multi-turn evaluation framework to test spoken dialogue models’ emotional intelligence beyond single-turn audio QA.

## What problem does it solve?

- Existing spoken-dialogue-model (SDM) benchmarks largely emphasize single-turn audio understanding or “multi-turn” that is effectively concatenated single-turn prompts, leaving genuinely *interactive* multi-turn conversational ability under-evaluated.
- Emotional intelligence (emotion recognition, reasoning, support, and application) is a key real-world capability for SDMs, but prior suites cover it only partially and not in fully interactive settings.

## What is the core method / protocol?

- Introduces **Multi-Bench**, positioned as the first SDM benchmark explicitly targeting **multi-turn interactive dialogue** with an emphasis on **emotional intelligence**.
- **Hierarchical structure**:
  - **Basic track**: emotion understanding + emotion reasoning.
  - **Advanced track**: emotion support + emotion application.
- **Five tasks** and ~**3.2K** samples; spans emotion recognition → complex reasoning → interactive dialogue.
- Provides a **reproducible evaluation framework** and reports experiments across “eight subsets” of the benchmark.

## What are the key metrics?

- Primarily benchmark-task performance (reported as “good” on basic tasks vs weaker on advanced interactive/reasoning tasks); the paper emphasizes that many prior works over-rely on *text-only* scoring and narrow recall-style metrics.
- (From the introduction) highlights evaluation axes often missing in prior work: genuine interaction, multi-turn context dependence, and assessed modalities (text vs speech).

## What are the main results?

- Evaluates **six representative SDMs**.
- Findings: SDMs perform relatively well on **basic understanding** tasks, but show substantial room for improvement on **advanced multi-turn interactive** dialogue and **reasoning-related** tasks, especially **emotion awareness** and **emotion application**.

## How is this similar to GALILEO?

- Shares the core motivation that **single-turn evaluation is insufficient** and that **multi-turn interaction** reveals qualitatively different failure modes.
- Useful as a neighboring citation for the claim “existing benchmarks often reduce multi-turn to concatenated single-turn; we need protocols that preserve interaction.”

## How is this different from GALILEO?

- Domain focus: **emotional intelligence** and **spoken dialogue models** (audio modality), rather than robustness to misleading/adversarial turns (e.g., sycophancy/persuasion/consistency attacks).
- Primary emphasis is on **capability evaluation** for EI, not explicitly on **robustness metrics** like time-to-failure / survival-style analyses.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO uses controlled multi-turn *perturbations* (pressure, misleading follow-ups, adversarial dialogue) and explicit robustness metrics, it can make a sharper causal claim about *stability under attack/pressure* than an EI capability benchmark.

## Where GALILEO is weaker / needs to improve

- If GALILEO is currently text-only, Multi-Bench is a reminder that **spoken modality** and paralinguistic cues matter for real dialogue agents; GALILEO’s framing should clarify whether we target text chat agents vs speech agents.

## Action items for GALILEO (experiments / method / writing)

- [ ] Related-work positioning: cite Multi-Bench as evidence that the community is moving toward **interactive multi-turn evaluation frameworks** (and that many earlier “multi-turn” benchmarks were not truly interactive).
- [ ] Consider adding a short paragraph distinguishing: (a) capability benchmarks (EI, task success) vs (b) robustness benchmarks (stability under pressure/adversarial multi-turn), and place GALILEO in (b).

## Quotes / details to potentially cite

- “We introduce Multi-Bench, the first benchmark explicitly designed to evaluate SDMs in multi-turn interactive dialogue with an emphasis on emotional intelligence.”
- “Results show that while current SDMs achieve good performance on basic understanding tasks, they still have room for improvement in advanced multi-turn interactive dialogue and reasoning-related tasks, particularly in emotion awareness and application.”
