# Towards Outcome-Oriented, Task-Agnostic Evaluation of AI Agents

- Year: 2025
- Venue: arXiv (white paper / industry report)
- Authors: Waseem AlShikh; Muayad Sayed Ali; Brian Kennedy; Dmytro Mozolevskyi
- URL: https://arxiv.org/html/2511.08242v1
- BibTeX key (if we add it): alshikh2025outcome
- Tags: agents, evaluation, outcome-oriented, task-agnostic, metrics, roi

## One-sentence takeaway

Proposes an outcome-oriented, task-agnostic suite of 11 metrics (goal completion, autonomy, resilience, tool dexterity, and business impact/ROI) and illustrates trade-offs across agent architectures via large-scale simulation.

## What problem does it solve?

- Argues that infra metrics (latency, throughput, token counts) and task-specific benchmarks do not capture the real utility of deployed agents.
- Targets the lack of a standardized, cross-domain evaluation framework that connects agent behavior to stakeholder outcomes and business value.

## What is the core method / protocol?

- Defines 11 metrics grouped into:
  - Performance & quality: Goal Completion Rate (GCR), Autonomy Index (AIx), Decision Turnaround Time (DTT), Cognitive Efficiency Score (CES), Tool Dexterity Index (TDI), Outcome Alignment Score (OAS), Collaboration Quality Index (CQI).
  - Resilience & adaptability: Multi-Step Task Resilience (MTR), Chain Robustness Score (CRS), Adaptability Delta (AD).
  - Economic impact: Business Impact Efficiency (BIE) (value per cost).
- Provides formulas plus a (synthetic) “large-scale experiment”:
  - 4 agent archetypes: ReAct, Chain-of-Thought, Tool-Augmented, Hybrid.
  - 5 business domains: healthcare, finance, marketing, legal, customer service.
  - Simulates thousands of tasks and reports aggregate metric comparisons and correlations.

## What are the key metrics?

- The paper’s contribution is the metric suite itself; notable definitions:
  - GCR: fraction of tasks achieving stakeholder-defined goal.
  - AIx: 1 - (human interventions / total task steps).
  - CES: (tokens + tool-call equivalents) / successful tasks (lower is better).
  - TDI: scored heuristic of tool usage quality (optimal vs misuse vs missed tool).
  - MTR: percent of multi-step tasks where agent self-recovers from initial errors.
  - BIE: business KPI value / operational cost.

## What are the main results?

- In their simulation, the “Hybrid” agent is most consistently strong across the majority of metrics (high GCR, autonomy, alignment, collaboration; good resilience) and achieves best ROI/BIE on average.
- Tool-augmented agents show strong efficiency/speed and tool dexterity, but may appear less autonomous/adaptable depending on assumptions.
- Highlights trade-offs: higher quality/outcome alignment can correlate with higher resource use (worse CES).

## How is this similar to GALILEO?

- Same motivation: evaluate agents based on outcomes/quality and not only system-level measures.
- Overlaps with GALILEO-style dimensions such as goal success, autonomy/human intervention, robustness, and tool-use quality.
- Useful as a “positioning” citation for why agent eval must include outcome and business-impact metrics.

## How is this different from GALILEO?

- Primarily a conceptual framework + formulas and a synthetic simulation study, not a benchmark with real tasks or a measurement/observability platform.
- Metrics are high-level and many require human judgment or business KPI instrumentation (OAS, CQI, BIE) rather than being automatically scoreable from traces.
- Does not provide standardized datasets, task suites, or reproducible environments (beyond claimed simulation code/data).

## Where GALILEO is stronger / cleaner (if true)

- Likely has clearer operational definitions tied to logged traces and real execution (function calls, tool traces, task success) rather than simulated distributions.
- Can evaluate on real-world tasks/benchmarks and provide more actionable debugging than aggregate outcome metrics.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently under-emphasizes economic/business impact, this paper is a reminder to connect eval to ROI-like metrics (even if only as optional reporting).
- May motivate better reporting of autonomy/human-intervention rates and resilience/self-recovery metrics.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding an “outcome-oriented metric taxonomy” paragraph that explicitly separates infra metrics vs outcome metrics, citing this paper as an industry-motivated framework.
- [ ] If feasible, add optional reporting hooks for business-impact proxies (cost per successful task; value-per-cost), clearly scoped as “deployment-side”.
- [ ] Map GALILEO’s existing metrics to (GCR/AIx/MTR/CRS/TDI) to show coverage and differentiators.

## Quotes / details to potentially cite

- “Evaluating their performance based solely on infrastructural metrics such as latency, time-to-first-token, or token throughput is proving insufficient.” (Abstract)
- Metric list (Section 3 / Table 1): GCR, AIx, DTT, CES, TDI, OAS, CQI, MTR, CRS, AD, BIE.
- Notes: evaluation study is explicitly simulated/synthetic (Section 4.3), so cite as a proposed framework rather than empirical truth.