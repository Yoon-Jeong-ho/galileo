# Bayesian Evaluation of Large Language Model Behavior

- Year: 2025
- Venue: NeurIPS 2025 Workshop on Evaluating the Evolving LLM Lifecycle: Benchmarks, Emergent Abilities, and Scaling (workshop paper)
- Authors: Shang Wu; Saatvik Kher; Catarina Belém; Padhraic Smyth
- URL: https://arxiv.org/abs/2511.10661
- BibTeX key (if we add it): wu2025bayesian
- Tags: evaluation; uncertainty-quantification; bayesian; stochastic-decoding; sequential-sampling; refusal; pairwise-preference

## One-sentence takeaway

A Bayesian framework for binary LLM behavior metrics (e.g., refusal / preference wins) that explicitly quantifies uncertainty from stochastic decoding and enables sequential sampling to reduce cost for a target confidence.

## What problem does it solve?

- Typical LLM “benchmark scores” (often aggregate binary judgments) are reported as point estimates and ignore uncertainty.
- A key neglected uncertainty source: LLMs are often *deployed stochastically* (temperature / sampling), but evaluations often use deterministic decoding or only 1 sample per prompt.
- For black-box/API evaluation, repeated sampling is expensive; we want uncertainty-aware sampling plans.

## What is the core method / protocol?

- Treat each benchmark item/prompt as generating a Bernoulli outcome under stochastic generation (e.g., refusal vs not, model-A preferred vs not).
- Use a Bayesian model to infer the underlying per-item and/or aggregate rates, producing posterior intervals/credible regions for metrics.
- Develop sequential sampling strategies (including Thompson sampling-style ideas) to decide which inputs to sample next to reduce uncertainty with fewer total generations.
- Case studies:
  1) Refusal rates on adversarial “harmful request” style prompts.
  2) Pairwise preference comparisons on open-ended interactive dialogue examples.

## What are the key metrics?

- Posterior distributions / credible intervals for:
  - Refusal rate (or harmful-response rate) aggregated over a benchmark.
  - Pairwise win-rate / preference probability between two LLMs.
- “Cost-effective” reduction in uncertainty (fewer samples for same uncertainty target) in sequential setting.

## What are the main results?

- Demonstrates that uncertainty due to stochastic generation can be material and should be reported, not just point estimates.
- Shows how Bayesian modeling produces interpretable uncertainty quantification for binary evaluation pipelines.
- Illustrates (in case studies) that sequential sampling can reduce the number of generations needed to reach a given confidence/uncertainty threshold.

## How is this similar to GALILEO?

- Shares the motivation that single-number evaluations can be misleading, especially when behavior varies across prompts and across runs.
- The “probabilistic behavior under repeated trials” framing matches GALILEO’s interest in stability/robustness across turns/pressure.

## How is this different from GALILEO?

- Focuses on *statistical estimation and uncertainty quantification* for (mostly) binary outcomes, not specifically multi-turn pressure protocols, belief drift, or persuasion tactics.
- Does not propose a new robustness benchmark; instead it proposes a measurement framework for existing benchmarks.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO provides a concrete multi-turn pressure protocol + task design, it addresses *what to measure* for robustness; this paper is mostly about *how to estimate uncertainty* for a given binary label pipeline.

## Where GALILEO is weaker / needs to improve

- If GALILEO reports only point estimates, it may be vulnerable to the critique here: ignoring uncertainty induced by stochastic decoding and finite sampling.
- If GALILEO uses deterministic decoding, results may not reflect deployed stochastic behavior (depending on intended deployment).

## Action items for GALILEO (experiments / method / writing)

- [ ] Add an “uncertainty reporting” section: for key rates (e.g., flip-rate / drift-rate / compliance under pressure), report confidence/credible intervals.
- [ ] If GALILEO uses stochastic decoding, explicitly incorporate repeated sampling per prompt/turn and aggregate with uncertainty.
- [ ] Consider a small sequential sampling experiment: allocate more samples to high-variance / ambiguous items to reduce CI width under a fixed budget.
- [ ] Clarify which uncertainty sources GALILEO captures (prompt sampling, model randomness, judge variance) and which it does not.

## Quotes / details to potentially cite

- Motivation: evaluations “often neglect statistical uncertainty quantification” and in particular uncertainty “induced by the probabilistic text generation strategies” used in deployment.
- Two example binary evaluation settings: (1) refusal rates on adversarial harmful prompts; (2) pairwise preferences in open-ended interactive dialogue.
