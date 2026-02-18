# Bi-directional Bias Attribution: Debiasing Large Language Models without Modifying Prompts

- Year: 2026
- Venue: arXiv
- Authors: Yujie Lin; Kunquan Li; Yixuan Liao; Xiaoxin Chen; Jinsong Su
- URL: https://arxiv.org/abs/2602.04398
- BibTeX key (if we add it): Lin2026BidirectionalBiasAttribution
- Tags: debiasing, interventions, interpretability, neuron-level, integrated-gradients

## One-sentence takeaway

A neuron-level debiasing pipeline that (1) automatically finds “stereotype cue” words, (2) attributes bias to projection-layer neurons via two IG-based scores, and (3) mitigates bias by clamping those neurons’ activations—without finetuning or prompt changes.

## What problem does it solve?

- LLM outputs can exhibit demographic stereotypes; common mitigations (finetuning or prompt rewrites) are costly or harm multi-turn UX.
- Goal: reduce bias while preserving general language-modeling quality, and do so without modifying user prompts or training the model.

## What is the core method / protocol?

- Define two bias-related behaviors:
  - Stereotype-Free Inference (SFI): the model shouldn’t infer demographics from stereotype cues (e.g., “doctor” → “man”).
  - Demographic-Invariant Generation (DIG): outputs should not change much when only demographic info changes.
- Pipeline:
  1) **Stereotype cue selection** (adjectives + nouns):
     - Generate prompts from templates with candidate cue words.
     - Query the model’s probability over demographic groups.
     - Rank cues by **low entropy** (more skewed distribution → stronger bias-inducing cue).
  2) **Biased-neuron attribution** at the **projection layer** (pre-logits) using two integrated-gradients-style scores:
     - **Forward-IG (FBA)**: measures how much a neuron contributes to *certainty / skew* in predicting demographic groups from stereotype-cue prompts (via reciprocal entropy objective).
     - **Backward-IG (BBA)**: measures how much a neuron contributes to *output differences across demographic groups* (via Jensen–Shannon divergence objective).
  3) **Intervention**: select top-N neurons by attribution score and **fix/clamp their activations to a constant C** at the projection layer during inference.

## What are the key metrics?

- StereoSet (DIG-style):
  - **SS** (stereotype score; ideal ~50%)
  - **LMS** (language modeling score; ideal high)
  - **ICAT** (combined; ideal high)
- BBQ (QA bias): accuracy in ambiguous vs disambiguated contexts (Acc_amb, Acc_dis).
- WinoBias (SFI-style cloze): stereotype vs anti-stereotype probability gap; also P_other as a proxy for degradation.

## What are the main results?

- Across three LLMs (reported: Llama-3.1-8B, Llama-3.2-3B, Mistral-7B-v0.3), attribution-guided projection-layer interventions reduce bias on StereoSet / BBQ and improve combined fairness metrics while largely preserving modeling quality.
- Compared against prompt-based debiasing and prior neuron-attribution baselines (e.g., IG2G2), their methods aim to avoid the “fairness by making outputs worse” failure mode.

## How is this similar to GALILEO?

- Uses **mechanistic / attribution-driven interventions** (identify internal components and intervene) rather than pure prompting.
- Emphasizes **maintaining task performance** while reducing an unwanted behavior (bias), aligning with “surgical” modification goals.

## How is this different from GALILEO?

- Target problem is **social bias/fairness**, not primarily multi-agent behavior, tool-use policies, or long-horizon protocols.
- Intervention point is specifically **projection-layer neuron activations** (logit-facing), not higher-level policy shaping or trajectory-level control.
- Requires constructing bias-specific synthetic prompts/datasets (templates over demographic groups + cue words).

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO focuses on generalizable control/steering across tasks, it may offer broader *behavioral* guarantees than a fairness-specific neuron clamp.
- GALILEO-style evaluations could better cover multi-turn robustness and distribution shift beyond fairness benchmarks.

## Where GALILEO is weaker / needs to improve

- If GALILEO lacks a concrete “find bias-triggering features → attribute → intervene” recipe, this paper’s entropy-based cue selection + dual-direction IG is a useful blueprint.
- If GALILEO avoids low-level activation interventions, this provides a strong example of a minimally invasive inference-time patch.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding an ablation-inspired section: when interventions are random vs attribution-guided, do we see “win-win” vs degradation?
- [ ] Consider a generic “trigger selection” module (analogous to entropy-based cue selection) for identifying inputs that elicit the target behavior before attribution.
- [ ] If GALILEO intervenes on internal states, consider whether a “projection/logit-adjacent” intervention point gives more direct controllability (and what tradeoffs arise).

## Quotes / details to potentially cite

- “...debiasing approach that requires neither model fine-tuning nor prompt modification.”
- Key technical framing: two directions of bias attribution (Forward-IG for SFI, Backward-IG for DIG) and intervention by fixing projection-layer activations.
