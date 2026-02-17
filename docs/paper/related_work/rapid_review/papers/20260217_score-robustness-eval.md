# SCORE: Systematic COnsistency and Robustness Evaluation for Large Language Models

- Year: 2025
- Venue: arXiv
- Authors: Grigor Nalbandyan, Rima Shahbazyan, Evelina Bakhturina
- URL: https://arxiv.org/abs/2503.00137
- BibTeX key (if we add it): nalbandyan2025score
- Tags: robustness, consistency, evaluation, prompt-variation, non-greedy, choice-order

## One-sentence takeaway

SCORE is a unified *non-adversarial* robustness evaluation framework that reports **accuracy ranges** (not single numbers) plus a **consistency rate** under prompt paraphrases, sampling randomness, and answer-choice reordering.

## What problem does it solve?

- Standard LLM evaluation typically reports one “best prompt / best setup” accuracy per benchmark, which can hide large variance under benign changes (prompt wording, option order, temperature seed).
- Robustness studies exist but are fragmented; SCORE aims to centralize/standardize the evaluation recipe and reporting.

## What is the core method / protocol?

- Evaluate the same benchmarks repeatedly under multiple *benign* setups:
  - **Prompt robustness:** 10 semantically similar prompts (mix of CoT + non-CoT; different question placement).
  - **Non-greedy robustness:** temperature 0.7 with **5 random seeds** (fixed prompt).
  - **Choice-order robustness:** for MCQ datasets, permute the answer option order while keeping the prompt/question fixed.
- Benchmarks used (for cost reasons): **MMLU-Pro**, **AGIEval**, **MATH**.
- Uses **generative evaluation** (extract final option letter / final numeric answer), intended to match real interaction better than log-likelihood scoring.
- Provides code and a robustness leaderboard.

## What are the key metrics?

- Accuracy reported as **mean + [min, max] range** across setups.
- **Consistency Rate (CR):** pairwise agreement rate among all predictions for the same item across setups.
  - For MCQ: equality of chosen letter.
  - For MATH: symbolic equivalence check (via SymPy).

## What are the main results?

- Benign changes can cause sizable swings (motivating examples):
  - Prompt paraphrasing on MMLU-Pro: up to ~10% accuracy fluctuation.
  - Choice reordering on AGIEval: up to ~6.1% accuracy differences.
- Accuracy and consistency are correlated but not identical; higher accuracy (or narrow accuracy range) does not guarantee higher consistency.
- Model size alone is not a reliable proxy for robustness/consistency.

## How is this similar to GALILEO?

- Same core message: **single-number accuracy is insufficient**; we need stability/robustness metrics.
- Emphasizes *repeated evaluation under controlled perturbations* and reporting distributions/ranges.

## How is this different from GALILEO?

- SCORE is largely **single-turn / per-item repeated evaluation**; GALILEO’s focus is **multi-turn dynamics** (time-to-failure, drift vs revision, recovery trajectories).
- SCORE’s perturbations are primarily formatting/elicitation (prompt, order, sampling seed), not sustained social pressure or persuasion operators.

## Where GALILEO is stronger / cleaner (if true)

- GALILEO can claim richer trajectory-level phenomena (flip timing, recovery after flip, operator-specific vulnerabilities), whereas SCORE focuses on repeated single-query robustness.
- GALILEO’s “drift vs evidence-driven revision” control story is outside SCORE’s scope.

## Where GALILEO is weaker / needs to improve

- We should ensure our reporting matches the “range + consistency” clarity SCORE advocates (avoid only reporting mean outcomes).
- We may be under-emphasizing *benign* robustness knobs (prompt paraphrase, option order, sampling randomness) as baselines/controls alongside multi-turn pressure.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a small **SCORE-style appendix table**: for a fixed subset of items, report outcome ranges + CR under (i) prompt variants, (ii) seed variants, (iii) option order variants (where applicable).
- [ ] In writing, explicitly contrast **non-adversarial robustness variance** (SCORE) vs **interactive multi-turn pressure dynamics** (GALILEO) and state why both matter.
- [ ] Consider adding “CR” as an auxiliary metric alongside our preferred multi-turn metrics, to create an easy bridge to this literature.

## Quotes / details to potentially cite

- “Typical evaluations … report a single metric … representing the model’s best-case performance … overlooks model robustness and reliability in real-world applications.”
- Examples of benign-variance magnitudes: prompt paraphrasing on MMLU-Pro (~10% swing) and choice reordering on AGIEval (~6.1% swing).
- Task definition summary: prompt robustness (10 prompts), non-greedy robustness (T=0.7, 5 seeds), choice-order robustness (MCQ option permutations).
