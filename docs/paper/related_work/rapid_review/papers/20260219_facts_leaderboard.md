# The FACTS Leaderboard: A Comprehensive Benchmark for Large Language Model Factuality

- Year: 2025
- Venue: arXiv
- Authors: Jacovi et al. (large Google author list)
- URL: https://arxiv.org/abs/2512.10791
- BibTeX key (if we add it): facts_leaderboard_2025
- Tags: factuality, evaluation, benchmarking, leaderboards, LLM-judges, grounding, multimodal, search

## One-sentence takeaway

A maintained, Kaggle-hosted leaderboard that aggregates four judge-scored factuality benchmarks (multimodal, closed-book/parametric, search-based, and document-grounded) into a single “FACTS Score” to track overall factual reliability.

## What problem does it solve?

- “Factuality” evaluation is fragmented: prior benchmarks target only one setting (e.g., grounding to a document vs. closed-book QA vs. tool-use/search, etc.).
- Model comparisons are hard because strengths differ by setting; a single benchmark can be misleading.
- Public leaderboards are vulnerable to overfitting; they propose public/private splits with centralized evaluation.

## What is the core method / protocol?

- Defines an evaluation suite with four sub-leaderboards:
  - FACTS Multimodal: factual responses to image-based questions.
  - FACTS Parametric: closed-book factoid questions (world knowledge in parameters).
  - FACTS Search: information-seeking with a search API/tool.
  - FACTS Grounding (v2): long-form answers grounded in provided documents; uses improved judge models.
- Uses automated judge models (“autoraters”) for scoring; reports accuracy per task and averages across tasks.
- Maintained as an online suite with public + private splits; evaluation run via Kaggle to limit leakage/overfitting.

## What are the key metrics?

- Per-task accuracy (reported with 95% confidence intervals), averaged over public and private sets.
- “FACTS Score”: mean of the four sub-task accuracies.

## What are the main results?

- Presents a reference leaderboard for proprietary models; the key point is that rankings can vary substantially across the four components.
- Example (from Table 1 in the paper): top systems show different profiles (e.g., strong search vs. weaker multimodal), motivating the aggregated score.

## How is this similar to GALILEO?

- Both are evaluation-centric and rely on automated judging to scale assessment.
- Emphasizes robustness against gaming (private splits, centralized evaluation), which is relevant for any benchmark intended to drive progress.

## How is this different from GALILEO?

- This work is a benchmark/leaderboard suite for factuality across multiple scenarios; it is not a method for improving model factuality.
- Heavily centered on Kaggle-hosted evaluation + leaderboards and specific factuality task design.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO targets a narrower scientific question, it may offer more controlled experimental conclusions than a broad leaderboard.
- If GALILEO includes stronger ablations or causal analysis, it can go beyond “who wins” to “why”.

## Where GALILEO is weaker / needs to improve

- If GALILEO’s evaluation covers fewer factuality regimes (e.g., only grounding), this suite highlights missing dimensions (multimodal, parametric, search).
- Benchmark integrity measures (public/private splits, centralized evaluation) may be a gap if GALILEO relies on fully public test sets.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add an evaluation section mapping GALILEO’s setting(s) onto the four “factuality regimes” (grounded vs. parametric vs. search/tool-use vs. multimodal), explicitly stating what is in/out of scope.
- [ ] If feasible, create a small “multi-regime” eval bundle (even partial) so we can report robustness beyond a single regime.
- [ ] Consider benchmark-integrity practices: private test split, controlled evaluation harness, or delayed-release prompts.

## Quotes / details to potentially cite

- The suite “aggregat[es] the performance of models on four distinct sub-leaderboards … designed to provide a robust and balanced assessment of a model’s overall factuality.”
- Introduces “FACTS Score” as the average accuracy across the four tasks (with public+private averaging per task).
- Kaggle-hosted with public/private splits to “guard its integrity” (reduce overfitting).
