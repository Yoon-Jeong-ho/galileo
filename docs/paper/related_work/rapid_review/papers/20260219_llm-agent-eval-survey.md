# Evaluation and Benchmarking of LLM Agents: A Survey

- Year: 2025
- Venue: KDD 2025 (survey; arXiv)
- Authors: Mahmoud Mohammadi et al.
- URL: https://arxiv.org/abs/2507.21504
- BibTeX key (if we add it): Mohammadi2025LLMAgentEvalSurvey
- Tags: agents, evaluation, survey, taxonomy, reliability, safety, enterprise

## One-sentence takeaway

A taxonomy-driven survey of how to evaluate LLM agents, framing “what to evaluate” (behavior/capabilities/reliability/safety) vs “how to evaluate” (interaction modes/data/metrics/tooling/contexts), with emphasis on enterprise constraints.

## What problem does it solve?

- Agent evaluation is fragmented: LLM agents act in interactive, tool-using, long-horizon settings where classic single-turn LLM metrics (or deterministic software testing) are insufficient.
- Practitioners need a structured map of objectives, methods, and tooling to build systematic evaluations and compare agents.

## What is the core method / protocol?

- Survey + two-dimensional taxonomy:
  - Evaluation objectives (what):
    - Agent behavior (task completion, output quality, latency/cost)
    - Agent capabilities (tool use, planning/reasoning, memory, collaboration)
    - Reliability (consistency/robustness, error handling)
    - Safety & alignment (trustworthiness, compliance/security)
  - Evaluation process (how):
    - Interaction mode (static vs interactive)
    - Evaluation data (synthetic vs real; domain benchmarks)
    - Metric computation (automatic metrics, human eval, LLM-judge)
    - Evaluation tooling (instrumentation/analytics/leaderboards)
    - Evaluation contexts/environments (simulators vs open-world)
- Highlights enterprise-specific evaluation needs: role-based access, reliability guarantees, long-horizon interactions, compliance.

## What are the key metrics?

- Mostly categorizes common metrics rather than proposing new ones; examples called out include:
  - Task success / success rate (SR), pass rate, goal completion
  - Quality judgments (human/LLM-based)
  - Reliability/robustness metrics (consistency across runs, performance under perturbations/errors)
  - Operational metrics (latency, cost)
  - Safety/alignment checks (policy compliance, harmfulness, fairness)

## What are the main results?

- A consolidated framework for agent evaluation and a set of open challenges:
  - Need for more holistic + realistic evaluations for interactive/long-horizon settings
  - Scalability concerns (cost of human eval; need for robust automation)
  - Enterprise gaps (access control, auditability, compliance) are under-addressed in research benchmarks

## How is this similar to GALILEO?

- Directly overlaps with GALILEO’s focus on evaluation objectives + process/tooling for agentic systems.
- Mentions evaluation tooling/instrumentation/analytics as a key pillar (the kind of layer GALILEO provides).

## How is this different from GALILEO?

- This is a survey/taxonomy, not a concrete evaluation product/system.
- Focus is breadth and categorization; less about implementable end-to-end workflows, dataset curation, or specific UI/analytics design.

## Where GALILEO is stronger / cleaner (if true)

- Can operationalize the survey’s “evaluation tooling” category into reproducible pipelines (data capture, traces, dashboards, regression tracking).
- Can provide opinionated defaults for enterprise requirements (RBAC, audit logs, compliance reports).

## Where GALILEO is weaker / needs to improve

- Survey highlights many evaluation axes; GALILEO should ensure coverage beyond task success (esp. reliability/robustness and safety) and beyond static tests.
- Long-horizon + interactive evaluations are emphasized; product needs strong support for multi-turn protocols and failure analysis.

## Action items for GALILEO (experiments / method / writing)

- [ ] Use this taxonomy as a framing section in related work: explicitly map GALILEO features to (objectives, process) categories.
- [ ] Ensure our evaluation suite includes reliability/robustness + safety checks as first-class dimensions (not just quality/success).
- [ ] Add an “enterprise evaluation requirements” paragraph (RBAC, auditability, compliance) aligned with the survey’s discussion.

## Quotes / details to potentially cite

- Abstract-level framing: two-dimensional taxonomy: (1) objectives (behavior/capabilities/reliability/safety) and (2) process (interaction modes, data/benchmarks, metrics computation, tooling).
- Enterprise challenges called out: role-based access to data, reliability guarantees, dynamic/long-horizon interactions, compliance.
