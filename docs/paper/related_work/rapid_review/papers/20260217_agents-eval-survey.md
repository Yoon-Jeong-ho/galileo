# Evaluation and Benchmarking of LLM Agents: A Survey

- Year: 2025
- Venue: KDD ’25 (ACM SIGKDD) / arXiv
- Authors: Mahmoud Mohammadi; Yipeng Li; Jane Lo; Wendy Yip
- URL: https://arxiv.org/abs/2507.21504
- BibTeX key (if we add it): Mohammadi2025AgentEvalSurvey
- Tags: agents, evaluation, survey, multi-turn, reliability, safety, benchmarks, metrics, tooling

## One-sentence takeaway

A survey that proposes a 2D taxonomy for evaluating LLM agents—*what* to evaluate (behavior/capabilities/reliability/safety) and *how* to evaluate (interaction mode/data/metrics/tooling/context)—with emphasis on enterprise constraints.

## What problem does it solve?

- The agent-evaluation landscape is fragmented: many benchmarks/metrics exist, but there is no unified way to organize evaluation goals (end-user outcomes vs internal capabilities vs reliability vs safety) and the evaluation pipeline (interactive vs static, datasets, metrics, tool support).
- Standard single-model LLM eval does not capture agents’ long-horizon, tool-using, probabilistic, environment-coupled behavior.

## What is the core method / protocol?

- Not a new benchmark; instead, a taxonomy + survey structure:
  - **Evaluation objectives (what):**
    - Agent behavior (task completion, output quality, latency/cost)
    - Agent capabilities (planning/reasoning, tool use, memory/context retention, multi-agent collaboration)
    - Reliability (consistency, robustness to input/environment perturbations, error handling)
    - Safety & alignment (harm prevention, compliance, security, fairness)
  - **Evaluation process (how):**
    - Interaction mode (static vs interactive)
    - Evaluation data (synthetic vs real; domain benchmarks)
    - Metric computation (automatic metrics; human judging; LLM-as-judge)
    - Tooling (instrumentation/observability frameworks; leaderboards)
    - Evaluation contexts/environments (simulations vs open-world: web/APIs)
- Highlights enterprise-focused requirements: RBAC/data access, auditability/compliance, reliability guarantees, and long-horizon dynamics.

## What are the key metrics?

- Surveyed rather than introduced; mentions common agent-level metrics such as:
  - success rate / pass rate / goal completion
  - quality judgments (human or LLM-based)
  - reliability/robustness notions (consistency across runs/perturbations)
  - operational metrics (latency, cost)

## What are the main results?

- Consolidates and organizes prior work into a coherent framework; identifies gaps and open directions:
  - need for more holistic + realistic evaluations (beyond narrow tasks)
  - scalable evaluation pipelines/tooling
  - enterprise-grade evaluation for compliance/reliability and long-horizon interactions

## How is this similar to GALILEO?

- Shares the core concern: **evaluating agent behavior over multi-turn, tool-using interaction trajectories**, including reliability and safety.
- Useful framing for writing: positions GALILEO’s contribution inside “reliability” (robustness/consistency) and “evaluation process” (interaction mode + metrics computation).

## How is this different from GALILEO?

- This is a **survey/taxonomy**, not a concrete evaluation protocol/benchmark with new metrics.
- Does not focus specifically on multi-turn *sycophancy / social pressure / belief drift*; treats reliability/safety broadly.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO provides a crisp, reproducible multi-turn protocol + metrics (e.g., turn-of-failure/time-to-failure, displacement curves, recovery), that is more actionable than high-level taxonomies.

## Where GALILEO is weaker / needs to improve

- Enterprise deployment constraints called out here (RBAC, audit/compliance, reliability guarantees) suggest we should be explicit about:
  - instrumentation/trace collection
  - reproducibility and evaluation harness design
  - reporting latency/cost, not just accuracy/robustness

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a short “positioning via agent-eval taxonomy” paragraph: map GALILEO metrics to (objectives × process) cells.
- [ ] Add an “evaluation process” checklist in appendix: interaction mode, dataset construction, judge method, environment, tooling.
- [ ] Consider reporting operational metrics (cost/latency) alongside robustness metrics when feasible.

## Quotes / details to potentially cite

- Introduces a two-dimensional taxonomy: evaluation objectives (behavior/capabilities/reliability/safety) × evaluation process (interaction modes, datasets/benchmarks, metric computation, tooling, evaluation contexts).
- Enterprise-specific challenges emphasized: role-based access to data, reliability guarantees, long-horizon interactions, compliance.
