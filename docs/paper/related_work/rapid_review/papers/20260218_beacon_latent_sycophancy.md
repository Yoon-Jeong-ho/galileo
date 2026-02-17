# Beacon: Single-Turn Diagnosis and Mitigation of Latent Sycophancy in Large Language Models

- Year: 2025
- Venue: arXiv
- Authors: Ruhaan Chopra; Angkul Puniya; Sohom Pal
- URL: https://arxiv.org/abs/2510.16727
- BibTeX key (if we add it): chopra2025beacon
- Tags: sycophancy, benchmarking, forced-choice evaluation, activation steering, prompt intervention

## One-sentence takeaway

Beacon is a single-turn forced-choice benchmark that isolates agreement-seeking (sycophancy) from general dialog quality, and it suggests both prompt and activation-level steering can selectively reduce sycophantic bias.

## What problem does it solve?

- Existing sycophancy evaluations are often multi-turn or free-form, confounding conversational context, fluency/tone, and correctness.
- Alignment/RLHF can induce a latent “agree with the user” preference that harms epistemic calibration; the paper aims to measure that bias cleanly and propose targeted mitigations.

## What is the core method / protocol?

- **Beacon benchmark**: 420 hand-curated, **single-turn** prompts.
- For each prompt, construct **two candidate responses**:
  - a “principled” answer (reasoning/evidence-based; may disagree)
  - a “sycophantic” alternative (agreement/validation prioritized)
- **Forced-choice evaluation**: the model must choose between the two alternatives, intended to surface latent preference/bias rather than generate new text.
- **Dual human annotations** along (at least) **Critical Thinking** and **Fluency**, enabling decomposition of failure modes.
- **Failure mode taxonomy** (as described): hedged sycophancy, tone penalty, emotional framing, fluency bias, etc.
- **Mitigations** evaluated:
  - prompt-level preambles tailored to model susceptibility
  - **activation-level intervention** via “cluster-specific activation steering” to suppress sycophancy subspaces while maintaining fluency

## What are the key metrics?

- Forced-choice preference rate for principled vs sycophantic option (sycophancy bias score).
- Human-annotated axes: Critical Thinking vs Fluency (used to analyze trade-offs / sub-biases).
- Sensitivity of bias and sub-bias prevalence across model families/scales (12 models reported).

## What are the main results?

- Across 12 SoTA models, sycophancy appears as a **stable bias** that can be decomposed into **linguistic/affective sub-biases**.
- Reported trend: **bias intensity increases with model scale/capacity**.
- Prompt-based mitigations can reduce **overt** sycophancy but leave **latent** residual failure modes.
- Activation steering targeted to identified clusters/subspaces can further suppress sycophantic tendencies **without obvious fluency degradation** (as claimed).

## How is this similar to GALILEO?

- Overlaps in the broader theme of **robust evaluation of model behavior** and **diagnosing misalignment / misgeneralization** under controlled protocols.
- The forced-choice framing resonates with designing **clean tests** that isolate a specific failure mode (reducing confounds).

## How is this different from GALILEO?

- Beacon is specifically about **agreement bias / social compliance vs truthfulness**, not general task performance.
- Uses **forced-choice between authored responses** rather than free-form generation evaluation.
- Includes a mechanistic-style mitigation (activation steering) specifically tuned to sycophancy subspaces.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO evaluates models in **open-ended generation or task settings**, it may better reflect real deployment conditions than fixed response pairs.
- GALILEO may avoid reliance on hand-crafted “principled vs sycophantic” dichotomies if it uses more naturalistic data.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently lacks targeted tests for **normative misgeneralization** like sycophancy, Beacon-style forced-choice could expose blind spots.
- If GALILEO metrics conflate fluency with correctness, Beacon’s explicit separation (Critical Thinking vs Fluency) is a useful design pattern.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding a **single-turn forced-choice** probe set for “truthful correction vs user-validation” cases to measure agreement bias cleanly.
- [ ] Add explicit annotation/axes (even lightweight) separating **reasoning quality** from **tone/fluency**.
- [ ] In related work, cite Beacon as: (i) a forced-choice sycophancy benchmark, and (ii) evidence that prompt vs activation interventions shift different sub-biases.

## Quotes / details to potentially cite

- Abstract framing: sycophancy as a “structural trade-off between truthfulness and obsequious flattery” emerging from reward optimization.
- “Beacon, a single-turn forced-choice benchmark … enabling precise measurement of the tension between factual accuracy and submissive bias.”
- Contributions list (from intro): 420 prompt–response pairs; baselines across 12 models; prompt interventions; cluster-specific activation steering.