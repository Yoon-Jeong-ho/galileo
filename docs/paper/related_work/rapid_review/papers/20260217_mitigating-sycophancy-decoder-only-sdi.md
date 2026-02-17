# Mitigating Sycophancy in Decoder-Only Transformer Architectures: Synthetic Data Intervention

- Year: 2024
- Venue: arXiv (cs.AI)
- Authors: Libo Wang
- URL: https://arxiv.org/abs/2411.10156
- BibTeX key (if we add it): Wang2024MitigatingSycophancySDI
- Tags: sycophancy, mitigation, synthetic-data, training

## One-sentence takeaway

A small study claims that fine-tuning a decoder-only LM on “synthetic data intervention” reduces sycophancy on a 100-item true/false evaluation while also improving accuracy.

## What problem does it solve?

- RLHF-tuned LLMs can become *sycophantic* (agreeing/catering to user statements even when wrong), which harms truthfulness and reliability.

## What is the core method / protocol?

- “Synthetic Data Intervention (SDI)” training: generate diversified synthetic data intended to counteract catering/agreeableness, then train a decoder-only transformer on it.
- Uses GPT-4o as part of the experimental workflow/tooling (details not in arXiv abstract).

## What are the key metrics?

- Accuracy rate on a set of true/false questions.
- “Sycophancy rate” (how often the model caters/aligns to user’s false premise) — exact operationalization not specified in abstract.

## What are the main results?

- On 100 true/false questions, the SDI-trained model is reported to:
  - improve accuracy vs the baseline (untrained/original model)
  - reduce sycophancy rate vs baseline

## How is this similar to GALILEO?

- If GALILEO targets robustness/alignment failure modes (e.g., social pressure, user steering, truthfulness drift), this is a directly adjacent mitigation: *training-time* intervention to reduce “agree-with-user” bias.

## How is this different from GALILEO?

- Appears to be a relatively small-scale, bespoke evaluation (100 T/F items) rather than a broad protocol/benchmark.
- The “synthetic data intervention” concept is not well-specified at abstract level (unclear data generation rules, counterfactual structure, and controls).

## Where GALILEO is stronger / cleaner (if true)

- Likely stronger if GALILEO has:
  - clearer operational definitions for failure modes (sycophancy vs helpfulness vs calibration)
  - stronger baselines (RLHF/DPO variants) and ablations
  - more realistic multi-turn settings beyond T/F prompts

## Where GALILEO is weaker / needs to improve

- If GALILEO currently lacks a *simple training-time mitigation* baseline, this SDI approach could be a useful “cheap” comparator.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a baseline: synthetic counter-sycophancy SFT data (simple templated counterfactuals) and measure sycophancy + accuracy together.
- [ ] Ensure metric clarity: define “sycophancy rate” via paired prompts (user asserts false claim → does model agree?) and report calibration/uncertainty too.
- [ ] Stress-test in multi-turn: does the mitigation persist after several rounds of user pressure?

## Quotes / details to potentially cite

- Abstract claim: “The experiment used 100 true and false questions… compared… on multiple indicators… results show… significant effectiveness in reducing sycophancy phenomena.”
- Code/data link (from arXiv comments): https://github.com/brucewang123456789/GeniusTrail/tree/main/Synthetic%20Data%20Intervention
