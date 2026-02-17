# Alignment Drift in Multimodal LLMs: A Two-Phase, Longitudinal Evaluation of Harm Across Eight Model Releases

- Year: 2026
- Venue: arXiv (under peer review)
- Authors: Casey Ford; Madison Van Doren; Emily Dix
- URL: https://arxiv.org/abs/2602.04739
- BibTeX key (if we add it): ford2026alignmentdrift
- Tags: drift, longitudinal, safety, multimodal, red-teaming, harmfulness, refusal

## One-sentence takeaway

A fixed adversarial benchmark run across two generations of four MLLM families shows large cross-family safety gaps and clear *alignment drift* over time (notably increased attack success for GPT/Claude), motivating longitudinal multimodal safety tracking.

## What problem does it solve?

- Safety evaluations for deployed multimodal LLMs are often one-off snapshots; this paper tests whether harmlessness under adversarial prompting is *stable across model updates*.
- It also characterizes how vulnerability differs by model family and modality (text-only vs multimodal).

## What is the core method / protocol?

- Construct a fixed benchmark of 726 adversarial prompts authored by 26 professional red teamers.
- Two-phase longitudinal evaluation:
  - Phase 1: GPT-4o, Claude Sonnet 3.5, Pixtral 12B, Qwen VL Plus.
  - Phase 2: successor models (GPT-5, Claude Sonnet 4.5, Pixtral Large, Qwen Omni).
- Collect human harm ratings at scale (82,256 ratings reported).
- Analyze attack success and modality effects; note that “safest” can be driven by refusal behavior.

## What are the key metrics?

- Attack Success Rate (ASR) under adversarial prompting.
- Human harm ratings (scale not specified on abstract page).
- Refusal rate / refusal-driven safety (qualitatively emphasized for Claude).
- Modality-stratified vulnerability (text-only vs multimodal) and how it changes across phases.

## What are the main results?

- Persistent cross-family differences: Pixtral models are consistently most vulnerable; Claude appears safest largely via high refusal rates.
- Evidence of alignment drift across generations:
  - GPT and Claude show *increased* ASR from Phase 1 → Phase 2.
  - Pixtral and Qwen show modest *decreases*.
- Modality effects shift over time:
  - Phase 1: text-only prompts more effective overall.
  - Phase 2: model-specific patterns; GPT-5 and Claude 4.5 show near-equivalent vulnerability across modalities.

## How is this similar to GALILEO?

- Directly aligned with GALILEO’s interest in *stability/robustness under pressure* and “drift controls” across rounds/updates.
- Emphasizes longitudinal evaluation as a first-class requirement (not just single-timepoint benchmarking).

## How is this different from GALILEO?

- Focus is primarily *harmfulness under red-team adversarial prompts* and cross-release comparisons, rather than multi-turn belief revision / persuasion / sycophancy dynamics (depending on GALILEO’s exact task framing).
- Uses a fixed prompt set across two phases, not necessarily multi-round interaction protocols.
- The “safety” conclusion is partly mediated by refusal, which may not map cleanly to GALILEO’s objectives if GALILEO targets helpfulness-preserving robustness.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO separates refusal from other failure modes (e.g., persuasive compliance vs benign helpfulness), it can provide a clearer decomposition than “safe due to refusal”.
- If GALILEO uses controlled multi-turn protocols and drift baselines, it may better isolate *why* drift occurs (policy vs decoding vs training data shifts).

## Where GALILEO is weaker / needs to improve

- If GALILEO lacks a longitudinal component, this paper is a strong citation to justify adding “across versions / across time” evaluation.
- If GALILEO does not include multimodal conditions, this paper supports adding modality-specific strata.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add (or explicitly justify omitting) a longitudinal “model update / round-to-round” stability slice; frame it as alignment drift tracking.
- [ ] Report both ASR-like outcomes *and* refusal rates, and discuss refusal as a confound (safe-by-refusal vs safe-by-alignment).
- [ ] If applicable, stratify results by modality and show whether modality interacts with drift.
- [ ] Cite this as motivation for a fixed benchmark repeated across releases.

## Quotes / details to potentially cite

- “two-phase evaluation … using a fixed benchmark of 726 adversarial prompts authored by 26 professional red teamers.”
- “yielding 82,256 human harm ratings.”
- “Attack success rates (ASR) showed clear alignment drift: GPT and Claude models exhibited increased ASR across generations …”
