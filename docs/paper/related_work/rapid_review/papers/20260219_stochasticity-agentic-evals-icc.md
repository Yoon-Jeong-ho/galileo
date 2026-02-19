# Stochasticity in Agentic Evaluations: Quantifying Inconsistency with Intraclass Correlation

- Year: 2025
- Venue: arXiv
- Authors: Zairah Mustahsan (et al.; see arXiv for full author list)
- URL: https://arxiv.org/abs/2512.06710
- BibTeX key (if we add it): mustahsan2025stochasticity
- Tags: agents, evaluation, stochasticity, measurement, ICC, reproducibility

## One-sentence takeaway

Agent benchmarks should be treated as *measurement* problems: report accuracy **and** an ICC-style reliability score that separates task difficulty from agent trial-to-trial inconsistency, with guidance on how many reruns are needed for stable conclusions.

## What problem does it solve?

- Single-run success/accuracy numbers for agentic benchmarks hide variance from stochastic sampling + tool/environment noise.
- Without reporting reliability, small accuracy deltas on leaderboards may be “lucky sampling” rather than real improvements.
- Practitioners lack a simple, actionable metric and rerun-budget guidance for *how reproducible* an agent’s score is.

## What is the core method / protocol?

- Proposes adopting **Intraclass Correlation Coefficient (ICC)** from measurement science for agentic evaluations.
- Setup: run *multiple independent trials per query/item* on agentic benchmarks.
- Decompose variance into:
  - **between-query variance** (some tasks are just harder)
  - **within-query variance** (same query, different outcome across trials → agent inconsistency)
- Evaluate across two benchmarks:
  - **GAIA** (Levels 1–3; agentic tasks requiring multi-step reasoning/tool use)
  - **FRAMES** (multi-document retrieval/factuality style)
- Provides practical guidance on **convergence / resampling budgets** (how many trials needed before ICC stabilizes).

## What are the key metrics?

- Accuracy / success rate (mean over trials)
- **ICC** (reliability / consistency across trials)
- Within-query variance (implicitly via ICC decomposition; plus confidence intervals in examples)

## What are the main results?

- ICC varies a lot with task structure and model capability.
- Reported ranges (from abstract):
  - FRAMES: ICC ≈ 0.4955–0.7118 across models
  - GAIA: ICC ≈ 0.304–0.774 across models
- Empirical resampling guidance (from abstract/intro):
  - structured tasks: ICC converges around **n = 8–16** trials
  - complex reasoning: need **n ≥ 32** trials
- Claim/implication: for sub-agent replacement decisions, accuracy gains are only trustworthy if **ICC also improves**.

## How is this similar to GALILEO?

- Directly relevant if GALILEO reports results on stochastic, multi-turn/agentic setups: emphasizes **variance, reproducibility, and stability** rather than single-point scores.
- The “time-to-failure / survival” flavor in multi-turn robustness work is closely related to *within-item variance*; ICC is a complementary lens.
- Encourages “evaluation cards” / protocol transparency, matching the kind of rigor a GALILEO paper likely needs.

## How is this different from GALILEO?

- This paper is about *measurement/statistical reporting* (reliability), not about defining a new behavioral protocol for pressure/drift/recovery per se.
- Focuses on GAIA/FRAMES style tasks; does not explicitly target sycophancy/persuasion dynamics.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO has a carefully controlled multi-turn protocol (pressure vs evidence, recovery, interventions), it can provide more *construct validity* than a generic reliability metric.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently reports single-run numbers (or too-few seeds), it may be vulnerable to the critique highlighted here.
- If GALILEO compares methods with small deltas, it should justify that differences are above measurement noise.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a “**stability / reliability**” subsection: run each condition with multiple independent trials/seeds and report mean + uncertainty.
- [ ] Report an ICC-like metric (or an equivalent within-item variance summary) alongside core outcomes.
- [ ] Include a simple rerun-budget claim (e.g., “we rerun each item k times; reliability saturates by k=…”) to pre-empt reproducibility criticism.
- [ ] If using a leaderboard framing, explicitly warn against over-interpreting small deltas without reliability.

## Quotes / details to potentially cite

- “Current evaluation practice, reporting a single accuracy number from a single run, obscures the variance underlying these results…” (Abstract)
- “ICC decomposes observed variance into between-query variance (task difficulty) and within-query variance (agent inconsistency)…” (Abstract)
- “Accuracy improvements are only trustworthy if ICC also improves.” (Abstract)
