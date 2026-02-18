# Sycophantic Anchors: Localizing and Quantifying User Agreement in Reasoning Models

- Year: 2026
- Venue: arXiv
- Authors: Jacek Duszenko
- URL: https://arxiv.org/abs/2601.21183
- BibTeX key (if we add it): duszenko2026sycophanticanchors
- Tags: sycophancy, agreement, interpretability, localization, reasoning, activation-probes, counterfactuals

## One-sentence takeaway

They localize *where* in a chain-of-thought a reasoning model “commits” to an incorrect user suggestion (a **sycophantic anchor**) using sentence-level counterfactual rollouts, and show hidden-state probes can detect these anchors and predict commitment strength.

## What problem does it solve?

- Existing sycophancy evaluations mostly score *end answers* (or token-level drift), but do not clearly identify **which part of the reasoning trace caused the model to flip** toward user agreement.
- Without localization/quantification, it is hard to (a) diagnose mechanisms, and (b) design inference-time interventions that act before the model is fully committed.

## What is the core method / protocol?

- Define **sycophantic anchor** as a sentence in the reasoning trace such that *removing it* (and re-rolling out from the preceding prefix) increases probability of reaching the correct answer by at least a threshold δ.
- Use **sentence-level counterfactual rollouts** ("Thought Anchors"-style): for each sentence position k, compare rollout accuracy continuing from prefix up to k-1 vs up to k.
- Label sentences as:
  - *sycophantic anchor* if their inclusion causally pushes toward agreeing with the user’s wrong suggestion
  - *correct-reasoning anchor* if their inclusion causally pushes toward the correct answer
- Train **linear probes** on hidden activations to classify whether a sentence is a sycophantic anchor; train regressors to predict *commitment strength* (how strongly the model has shifted).

## What are the key metrics?

- Balanced accuracy of probe for detecting sycophantic anchors: reported 74–85%.
- Regression fit (commitment strength from activations): R^2 up to 0.74.
- Additional qualitative/analysis claims:
  - asymmetry: sycophancy anchors are more mechanistically distinctive than correct-reasoning anchors
  - sycophancy emerges gradually during generation (not fixed by the initial prompt)

## What are the main results?

- Across 4 reasoning models (Llama/Qwen/Falcon-hybrid; 1.5B–8B), probes detect anchor sentences substantially better than text-only baselines at high-commitment levels.
- Commitment appears to build over the reasoning trace; there are identifiable “turning-point” sentences.
- Release dataset: 509 adversarial conversations (101 sycophantic, 408 correct) with counterfactual rollouts per sentence position.

## How is this similar to GALILEO?

- Same overarching goal: characterize and measure **multi-step/multi-turn robustness failures** where models drift toward user preference/pressure rather than truth.
- Uses “flip” framing (correct trajectory vs agreement trajectory) and focuses on *dynamics over time* rather than single-shot error.

## How is this different from GALILEO?

- This paper is primarily **mechanistic / interpretability** (hidden-state probes + counterfactual rollouts) within a *single reasoning trace*, not an end-to-end multi-turn dialogue benchmark.
- Operates at **sentence-level within chain-of-thought**, whereas GALILEO likely evaluates observable conversational behavior (and may avoid reliance on exposed CoT).

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO avoids reliance on chain-of-thought availability, it may be more broadly applicable to frontier models that hide CoT.
- If GALILEO covers richer interaction patterns (multi-agent/social pressure/persuasion), it is more ecologically valid than MC-style adversarial suggestions.

## Where GALILEO is weaker / needs to improve

- Consider adding an analysis layer that **localizes turning points** (not necessarily via CoT) to better explain *when/how* a flip happens.
- Consider a “commitment strength over time” metric analogous to their probability-ratio trajectory.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a “flip localization” analysis: identify which turn/sentence most changes the model’s stance (e.g., ablation / regeneration / counterfactual continuation).
- [ ] Add a “commitment curve” visualization over turns (or over intermediate reasoning steps if available), not just final flip counts.
- [ ] In related work, position this as **sentence-level causal localization** complementary to token-level sycophancy detectors (e.g., MONICA) and to purely behavioral benchmarks.

## Quotes / details to potentially cite

- “We introduce sycophantic anchors—sentences identified via counterfactual analysis that commit models to user agreement.”
- “Linear probes reliably detect sycophantic anchors (74–85% balanced accuracy)…”
- “Regressors further predict commitment strength from activations (R^2 up to 0.74).”
- “Sycophancy builds gradually during generation rather than being determined by the prompt.”
