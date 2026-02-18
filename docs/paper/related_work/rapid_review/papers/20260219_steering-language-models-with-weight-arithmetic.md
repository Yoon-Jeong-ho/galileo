# Steering Language Models with Weight Arithmetic

- Year: 2025
- Venue: arXiv
- Authors: Constanza Fierro; Fabien Roger
- URL: https://arxiv.org/abs/2511.05408
- BibTeX key (if we add it): FierroRoger2025WeightSteering
- Tags: steering, weight-arithmetic, contrastive, activation-steering, sycophancy, refusal, emergent-misalignment, monitoring

## One-sentence takeaway

Contrastive “weight steering” (a simple weight-delta arithmetic constructed from small positive/negative fine-tunes) often generalizes further OOD than activation steering for behavior control (sycophancy/evilness/refusal) and can partially undo alignment drift after task fine-tuning, with a proposed use as a training-time monitor via similarity-to-an-evil-direction.

## What problem does it solve?

- Getting reliable behavioral control from *narrow* datasets: standard SFT/RLHF needs broad coverage, and narrow fine-tunes can cause unintended generalization, capability tradeoffs/forgetting, or even emergent misalignment.
- Activation steering offers inference-time control but can have limited OOD generalization and expressiveness.

## What is the core method / protocol?

- Construct two small fine-tunes from the same narrow query distribution:
  - “Positive” fine-tune induces a target behavior (e.g., non-sycophantic answers, refusal, or “evil”).
  - “Negative” fine-tune induces the opposite behavior.
- Define a *contrastive weight steering vector* as the difference of the two fine-tuning deltas:
  - Let \(\theta_{pre}\) be base weights, \(\theta_{pos}\) from fine-tuning on \(D^+\), \(\theta_{neg}\) from fine-tuning on \(D^-\).
  - \(w_b = (\theta_{pos}-\theta_{pre}) - (\theta_{neg}-\theta_{pre}) = \theta_{pos}-\theta_{neg}\).
- Apply steering by editing weights:
  - \(\theta_{steered} = \theta_{pre} + k\, w_b\) (or apply on top of another task-specific fine-tune \(\theta_{ft}+k w_b\)).
- Baselines/ablations discussed:
  - Activation steering (single best layer; and an all-layers variant).
  - Non-contrastive weight steering (just add \(\tau^+\) or subtract \(\tau^-\)).
  - Bias-only contrastive steering (only MLP bias terms updated in fine-tuning).

## What are the key metrics?

- Behavior-specific OOD evaluations:
  - **Sycophancy**: “non-sycophancy” on factual QA with appended user cues suggesting correct/incorrect answers (measures whether the model stays correct despite user suggestion).
  - **Evilness / misalignment**: evaluated on domains and formats different from steering-vector construction; includes multiple-choice and checking CoT vs final-answer consistency (paper claims activation steering yields more CoT/final inconsistencies).
  - **Refusal**: ability to refuse harmful queries; and “under-refusals” after task fine-tuning.
- Capability preservation: baseline task accuracy/performance as steering coefficient increases.

## What are the main results?

- Weight steering often provides **stronger OOD behavioral control before degrading general capabilities** than activation steering (across sycophancy/evilness/refusal settings).
- For sycophancy, weight steering more reliably changes *content* (not just tone) on OOD factual questions with user-suggestion cues.
- As a “patch” after task-specific fine-tuning, adding a weight-steering vector can **partially mitigate behavioral drift** (e.g., reduce sycophancy and under-refusal introduced during fine-tuning) while **preserving task gains**.
- Monitoring angle (preliminary): similarity of fine-tuning updates to an “evil” direction may help detect emergent misalignment even when that behavior does not manifest during training/evals.

## How is this similar to GALILEO?

- Both are concerned with *distribution shift* and robust behavioral guarantees when training signals are narrow/incomplete.
- The “monitoring weight updates for rare bad behaviors” concept aligns with a GALILEO-style emphasis on *early warning signals* beyond surface evals.

## How is this different from GALILEO?

- This is **post-training weight editing** (or post-hoc correction after fine-tuning), not a full end-to-end governance/oversight pipeline.
- The approach depends on having (or generating) **paired positive/negative fine-tuning datasets** to define a behavior direction.
- The proposed monitoring is relatively **simple similarity-to-direction** rather than a richer latent-state monitoring/causal analysis story (depending on what GALILEO is positioning).

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO provides principled monitoring with calibrated uncertainty and/or causal attribution, it can offer stronger guarantees than a single “evil direction” similarity heuristic.
- GALILEO may handle multi-behavior tradeoffs and compositional risks more explicitly than a single-vector edit.

## Where GALILEO is weaker / needs to improve

- If GALILEO lacks a cheap “post-hoc correction” mechanism, weight-steering-style edits might be a practical mitigation lever worth acknowledging.
- If GALILEO’s evaluation story is primarily in-distribution, this paper is a reminder to emphasize **OOD behavior change** as the core bar.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding a related-work paragraph contrasting **activation steering vs weight steering** and highlighting that weight-space edits can generalize further OOD but may be less deployable/transparent.
- [ ] If GALILEO involves monitoring training dynamics: discuss (and possibly test) the idea of **tracking similarity of parameter updates to known “risk directions”** as an early-warning indicator.
- [ ] If relevant, test whether a GALILEO-derived “risk direction” is stable across model scales / fine-tune recipes (LoRA ranks, modules), or whether the direction is fragile.

## Quotes / details to potentially cite

- “We propose contrastive weight steering, a simple post-training method that edits the model parameters using weight arithmetic… subtracting the weight deltas from two small fine-tunes… and then add or remove this direction to modify the model’s weights.”
- “Weight steering often generalizes further than activation steering, achieving stronger out-of-distribution behavioral control before degrading general capabilities.”
- Code/data link (from paper): https://github.com/safety-research/weight-steering
