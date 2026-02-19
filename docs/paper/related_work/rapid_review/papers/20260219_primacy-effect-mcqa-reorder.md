# Exploiting Primacy Effect To Improve Large Language Models

- Year: 2025
- Venue: RANLP 2025 (accepted; arXiv)
- Authors: Bianca Raimondi, Maurizio Gabbrielli
- URL: https://arxiv.org/abs/2507.13949
- BibTeX key (if we add it): Raimondi2025PrimacyEffect
- Tags: primacy-bias, positional-bias, MCQA, option-ordering, evaluation, training-free

## One-sentence takeaway

Fine-tuning amplifies primacy bias in MCQA, and simply reordering answer options by semantic similarity to the query (training-free) can exploit this bias to improve accuracy.

## What problem does it solve?

- MCQA accuracy of LLMs is highly sensitive to the ordering of answer options (positional bias: primacy/recency), which can cause brittle and unfair evaluation.
- Rather than only “mitigating” the bias, they ask: can we *use* the bias to improve performance without knowing the gold label?

## What is the core method / protocol?

- **Bias measurement protocol:** For each query, systematically shuffle the *correct* label across positions; record accuracy vs. position to quantify primacy/recency effects.
- **Claim about training:** Compare pre-trained vs. fine-tuned variants; report that fine-tuning strengthens positional bias.
- **Training-free reordering technique (main):**
  - Compute semantic similarity between the query and each candidate option (label).
  - **Primacy-exploitation:** sort options in *descending* similarity so “most relevant” options appear early.
  - **Recency-exploitation:** also consider *ascending* order (push relevant options late), motivated by recency.
  - Consider a combination of the two (details not fully visible from the truncated HTML extract, but the paper states they combine techniques).
- Evaluated on intent-classification-as-MCQA style datasets with a fixed label set per dataset.

## What are the key metrics?

- Primary: **MCQA accuracy** (correct label selected).
- Diagnostic: **accuracy as a function of correct-option position** (primacy/recency curves).

## What are the main results?

- Reordering options by similarity yields **significant accuracy gains** in MCQA.
- Gains are especially pronounced for **fine-tuned models**, consistent with their stronger primacy bias.
- Effects are reported as consistent across multiple datasets and model architectures (per abstract/intro).

## How is this similar to GALILEO?

- Both are about **evaluation artifacts and model behavior quirks** that can masquerade as “capability” (here: option-order bias; in GALILEO: multi-round degradation / failure dynamics).
- Both motivate **bias-aware evaluation design** and emphasize that protocol details can materially change conclusions.

## How is this different from GALILEO?

- This paper is **single-turn MCQA** with many candidate labels/options; GALILEO is about **multi-round interaction** (survival / turn-of-failure / flips) rather than positional choice among fixed options.
- Their intervention is **input reordering of answer choices**; GALILEO’s interventions (as framed in the project) are more about **robustness across rounds** and analyzing failure trajectories.

## Where GALILEO is stronger / cleaner (if true)

- GALILEO’s survival/TOF framing targets **temporal robustness** and can reveal dynamics that are not visible in single-turn accuracy.
- GALILEO can treat “flip” events as first-class objects and audit them with taxonomy / examples (more explanatory than raw positional curves).

## Where GALILEO is weaker / needs to improve

- If any GALILEO experiments rely on **multiple-choice style prompts** (or candidate lists), results could be confounded by **primacy/recency** unless order is controlled/randomized.
- If GALILEO uses “choose from the following” formats anywhere, it should explicitly document option ordering policy.

## Action items for GALILEO (experiments / method / writing)

- [ ] Audit GALILEO prompts for any **explicit candidate lists**; if present, add **order randomization / multiple order seeds** or a documented canonical ordering.
- [ ] Add a short related-work paragraph on **positional bias in LLM evaluation (primacy/recency)** and how protocol choices can change measured performance.
- [ ] If GALILEO includes MCQA-style baselines, consider reporting **order-sensitivity** as a robustness axis (analogous to round-sensitivity).

## Quotes / details to potentially cite

- Abstract (problem/solution framing): positional biases (primacy/recency) influence MCQA accuracy; fine-tuning amplifies primacy bias; reordering options by semantic similarity (response-independent) improves performance.
- Method sketch (from intro/figures): they evaluate by fixing the query and systematically moving the target label across positions, observing correctness concentrated in early positions (primacy).
