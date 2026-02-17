# Beyond Consensus: Mitigating the Agreeableness Bias in LLM Judge Evaluations

- Year: 2025
- Venue: arXiv
- Authors: Suryaansh Jain; Umair Z. Ahmed; Shubham Sahai; Ben Leong
- URL: https://arxiv.org/abs/2510.11822
- BibTeX key (if we add it): beyond-consensus-agreeableness-bias-2025
- Tags: agreeableness-bias, llm-as-judge, evaluation-bias, robustness

## One-sentence takeaway

LLM-as-a-judge evaluators are strongly *positively biased* (high TPR but very low TNR), so naive majority-vote ensembles can still overestimate quality; the paper proposes a minority-veto ensemble and a small-human-calibration regression to correct judge bias.

## What problem does it solve?

- Automated evaluation for open-ended outputs where correctness is subjective/semantic and there is no cheap verifier (their motivating case: feedback on buggy student code).
- Diagnoses a specific failure mode of LLM judges: **agreeableness / positivity bias** leading to **very low true-negative rates** (difficulty calling outputs *invalid*), which inflates perceived generator precision—especially under class imbalance.

## What is the core method / protocol?

- Task setting: generate natural-language feedback for buggy Python programs; humans label each feedback item as valid/invalid (with finer subcategories).
- Treats each LLM both as a **generator** (producing feedback) and a **validator** (judging feedback validity).
- Key proposals:
  - **Minority-veto** ensemble rule: instead of majority vote, allow a small “minority” of judges to veto validity (intended to counteract overly-permissive validators and handle missing judge outputs).
  - **Regression-based calibration**: use a small set of human-labeled generators to jointly estimate (i) generator precision and (ii) validator reliability (TPR/TNR), then correct bias when estimating new generators.

## What are the key metrics?

- Validator quality: **TPR** (detect valid) and **TNR** (detect invalid).
- Generator quality: **precision** (fraction of feedback labeled valid).
- Estimation accuracy: **maximum absolute error** between estimated vs true generator precision (under varying ensemble/calibration strategies).

## What are the main results?

- Empirical bias is severe: validators can have **TPR >96%** but **TNR typically <25%**, meaning they rarely flag invalid outputs.
- Majority voting helps but is insufficient (and sensitive to missing validator outputs).
- Regression calibration with limited human ground truth reportedly reduces maximum absolute error to **~1.2%**, about **2×** better than the best 14-model ensemble baseline on their dataset.

## How is this similar to GALILEO?

- Directly relevant to **evaluation reliability**: if GALILEO uses any LLM-judge components (or reports results that depend on automated auditing), this paper provides a concrete, quantifiable warning that judges can be **asymmetric** (good at confirming “valid” but bad at catching “invalid”).
- Fits the broader theme of **bias under social/interactional framing**: “agreeableness” is an evaluator-side analogue of sycophancy.

## How is this different from GALILEO?

- This is primarily about **measurement bias in evaluators**, not about multi-turn belief drift / recovery dynamics in assistants.
- Their empirical domain is open-ended program-feedback validation (single-step items), not persuasion/social-pressure trajectories.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO’s evaluations rely on **paired controls** (neutral vs pressure) and/or human-verified slices, that provides a cleaner causal story than purely judge-driven scoring.

## Where GALILEO is weaker / needs to improve

- If we use LLM-as-a-judge for any “invalid / unsafe / incorrect” detection, we may be **overstating robustness** due to low judge TNR (false negatives on “bad” outputs).

## Action items for GALILEO (experiments / method / writing)

- [ ] When using automated judges, always report **TPR/TNR (or sensitivity/specificity)** on a small human-labeled audit set; do not report only an aggregate accuracy/score.
- [ ] Consider a **veto-style aggregation** (or otherwise asymmetric thresholding) for detecting failures, since “calling something wrong” appears harder for judges.
- [ ] Add an “evaluation bias” limitation paragraph: low TNR under class imbalance can inflate reliability and obscure rare-but-important failures.

## Quotes / details to potentially cite

- Abstract: LLM judges can have **True Positive Rate ~96%** but **True Negative Rate <25%**, causing inflated reliability under class imbalance.
- Intro contribution claim: large-scale analysis of LLMs as judges for a subjective task; proposes minority-veto and a regression calibration using a small amount of ground-truth human labels.
