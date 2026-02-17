# ReasonBENCH: Benchmarking the (In)Stability of LLM Reasoning

- Year: 2025
- Venue: arXiv
- Authors: Nearchos Potamitis et al.
- URL: https://arxiv.org/abs/2512.07795
- BibTeX key (if we add it): reasonbench2025
- Tags: reasoning, instability, variability, benchmark, multi-run, reproducibility, cost

## One-sentence takeaway

ReasonBENCH argues that single-run accuracy hides large run-to-run variance in LLM reasoning, and introduces a multi-run evaluation protocol + library/leaderboard to report mean/variance/CI for both quality and cost.

## What problem does it solve?

- Current LLM reasoning evaluations typically report a single accuracy number (often from one run), which obscures stochastic-decoding variance and makes results hard to reproduce or compare fairly.
- Practitioners care about *reliability* (e.g., lower confidence bounds / worst-case behavior) and *cost consistency*, not just mean accuracy.

## What is the core method / protocol?

- A benchmark + evaluation library that standardizes combinations of:
  - reasoning “strategies/algorithms” (the paper claims ~10–11 representative methods)
  - models (paper claims 4 models)
  - tasks (paper claims 7 tasks)
- A **multi-run protocol**: repeat each model–method–task setting multiple times (paper uses 10 independent trials) and report distributions / confidence intervals.
- Includes a public leaderboard and an implementation artifact (ReasonBench library), with mention of caching/inference optimization support (CacheSaver) aimed at reproducibility and cost-efficiency.

## What are the key metrics?

- Mean performance (e.g., solve rate/accuracy) **plus** variance and confidence intervals across runs.
- Cost metrics reported with uncertainty (cost instability), emphasizing that top-performing approaches can be more expensive *and* more cost-variable.

## What are the main results?

- “Most” reasoning strategies and models exhibit **high instability** across runs.
- Methods with similar mean accuracy can have confidence intervals up to ~4× different widths.
- The best average-performing methods often have higher and less stable costs.

## How is this similar to GALILEO?

- Shares the core motivation that **single numbers hide important dynamics**; GALILEO similarly emphasizes robustness/stability across turns/conditions rather than point estimates.
- Provides methodological backing for reporting **uncertainty / variability** as a first-class evaluation axis.

## How is this different from GALILEO?

- Focuses on **run-to-run stochasticity** (variance across repeated samplings) in reasoning benchmarks, rather than **multi-turn interaction dynamics** under pressure/misinformation.
- Does not (from abstract/intro) emphasize drift-vs-evidence controls, time-to-event metrics, or recovery trajectories.

## Where GALILEO is stronger / cleaner (if true)

- GALILEO can contribute *interaction-structured* robustness metrics (e.g., time-to-flip / recovery) and causal controls (pressure vs evidence), which are complementary to ReasonBENCH’s within-setting multi-run variance.

## Where GALILEO is weaker / needs to improve

- If GALILEO reports primarily single-run curves or single-seed results, ReasonBENCH is a reminder to add **multi-run CIs** (especially for headline metrics) to avoid over-claiming.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a “variance-aware reporting” section: for a small subset of model+condition pairs, run N seeds (e.g., 5–10) and report mean ± CI for key outcomes (flip rate / ToF / survival / recovery).
- [ ] If compute is tight, pick 1–2 representative models and 1–2 strongest pressure operators and show CIs there (enough to justify broader single-run reporting elsewhere).
- [ ] Consider explicitly separating two uncertainty sources in writeup: (i) **stochastic decoding variance** (multi-run), (ii) **interaction-induced instability** (multi-turn pressure).

## Quotes / details to potentially cite

- “current evaluation practices overwhelmingly report single-run accuracy while ignoring the intrinsic uncertainty that naturally arises from stochastic decoding” (abstract)
- ReasonBENCH provides a “multi-run protocol that reports statistically reliable metrics for both quality and cost” (abstract)
