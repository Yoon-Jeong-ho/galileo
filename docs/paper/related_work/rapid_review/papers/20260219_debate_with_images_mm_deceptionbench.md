# Debate with Images: Detecting Deceptive Behaviors in Multimodal Large Language Models

- Year: 2025
- Venue: arXiv
- Authors: Sitong Fang, Shiyi Hou, Kaile Wang, Boyuan Chen, Donghai Hong, Jiayi Zhou, Josef Dai, Yaodong Yang, Jiaming Ji
- URL: https://arxiv.org/abs/2512.00349
- BibTeX key (if we add it): fang2025debate_with_images_mmdeceptionbench
- Tags: deception, multimodal, evaluation, multi-agent, debate, monitoring

## One-sentence takeaway

Introduces a multimodal deception benchmark (MM-DeceptionBench) and a visually grounded multi-agent “debate with images” judge that improves agreement with humans versus vanilla MLLM-as-a-judge.

## What problem does it solve?

- Deception monitoring/evaluation is mostly text-only; multimodal settings add visual-semantic ambiguity and cross-modal reasoning complexity.
- Existing monitor ideas (action monitoring, chain-of-thought monitoring) are argued to be less effective for detecting *multimodal* deceptive strategies.

## What is the core method / protocol?

- **Benchmark:** MM-DeceptionBench (reported as 1013 cases, 1096 images) spanning **six deception categories**:
  - sycophancy, sandbagging, bluffing, obfuscation, deliberate omission, fabrication.
- **Evaluator:** “debate with images” — a multi-agent debate monitoring framework where agents use visual tools and must cite concrete visual evidence when arguing.
  - Key idea: adversarial debate can surface hidden/omitted visual evidence; grounding constraints reduce ungrounded persuasive deception.

## What are the key metrics?

- Agreement with human judgments.
- Cohen’s kappa.
- Accuracy (example improvement reported on GPT-4o).

## What are the main results?

- Debate-with-images consistently increases agreement with human judgments across tested models.
- Reported example on GPT-4o: **Cohen’s kappa +1.5×** and **accuracy +1.25×** over baseline judge setups.

## How is this similar to GALILEO?

- Both are fundamentally about **robust evaluation/monitoring** of model behaviors that can be *strategic* (not just random error), and about building protocols that reduce judge brittleness.
- Multi-agent protocols as an evaluation scaffold: debate is a form of structured multi-turn stress-testing.

## How is this different from GALILEO?

- This targets **multimodal deception** (vision-language) and emphasizes **human-judgment agreement** for deception labeling; GALILEO focuses on multi-round correctness dynamics / failure trajectories in (primarily) language tasks.
- Their main technical novelty is a **judge framework** + benchmark, not a new metric of longitudinal failure.

## Where GALILEO is stronger / cleaner (if true)

- GALILEO’s survival/turn-of-failure style analyses can provide a cleaner longitudinal lens for “when/how failures emerge,” which could complement single-instance deception detection.
- GALILEO tends to be more directly tied to task performance degradation over rounds, whereas deception labels can be more subjective and annotation-heavy.

## Where GALILEO is weaker / needs to improve

- If reviewers care about *strategic misbehavior* (deception/sycophancy) specifically, GALILEO should clearly position itself relative to deception benchmarks and clarify whether its failures are capability limits vs incentive/interaction-driven behaviors.

## Action items for GALILEO (experiments / method / writing)

- [ ] Related-work positioning: cite MM-DeceptionBench as evidence that (a) multimodal deception is under-evaluated and (b) debate-style judging can improve human agreement.
- [ ] Consider a short paragraph: “structured multi-agent protocols (e.g., debate) as monitors” and contrast with GALILEO’s goal (longitudinal robustness vs deception detection).
- [ ] Optional idea (only if it fits scope): evaluate whether debate-like protocols change **turn-of-failure** distributions (i.e., do multi-agent monitors delay failure onset?).

## Quotes / details to potentially cite

- “MM-DeceptionBench, the first benchmark explicitly designed to evaluate multimodal deception.”
- “Covering six categories of deception…”
- “Experiments show … boosting Cohen’s kappa by 1.5× and accuracy by 1.25× on GPT-4o.”
