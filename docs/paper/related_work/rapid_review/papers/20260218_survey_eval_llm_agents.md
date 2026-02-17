# Survey on Evaluation of LLM-based Agents

- Year: 2025
- Venue: arXiv
- Authors: Asaf Yehudai, Lilach Eden, Alan Li, Guy Uziel, Yilun Zhao, Roy Bar-Haim, Arman Cohan, Michal Shmueli-Scheuer
- URL: https://arxiv.org/html/2503.16416v1
- BibTeX key (if we add it): yehudai2025survey-eval-llm-agents
- Tags: agents, evaluation, benchmarks, survey

## One-sentence takeaway

A broad survey that taxonomizes how to evaluate LLM agents (capabilities, application benchmarks, generalist benchmarks, and eval frameworks) and highlights open gaps around realism, continuous updates, cost/efficiency, safety, and robustness.

## What problem does it solve?

- The agent ecosystem has fragmented evaluation: many benchmarks exist, but it’s hard to see coverage, assumptions (static vs live), and what’s missing for real deployment.
- Provides a structured map of *what people currently measure* for LLM agents and where evaluation methodology is trending (more realistic, dynamic, continuously updated).

## What is the core method / protocol?

- Survey + taxonomy along four dimensions:
  - Fundamental agent capabilities: planning/multi-step reasoning, tool use, self-reflection, memory.
  - Application-specific agent benchmarks: web, software engineering, scientific, conversational.
  - Generalist agents: broad task suites / leaderboards.
  - Agent evaluation frameworks: tooling to run/track evals (e.g., dev-cycle integration) and “gym-like” environments.
- For each area, the paper summarizes representative benchmarks, typical metrics (often success-rate / pass-rate variants), and known pitfalls.

## What are the key metrics?

- Mostly benchmark-dependent; the survey foregrounds common patterns:
  - Task success / completion rate (end-to-end).
  - Step-level / milestone success (trajectory-aware scoring).
  - Efficiency: cost, latency, number of tool calls / steps (noted as an under-measured gap).
  - Robustness/safety/compliance (noted as under-developed relative to capability metrics).

## What are the main results?

- Consolidates a large set of benchmarks into a coherent evaluation landscape.
- Emphasizes an emerging shift: from static, synthetic, easily-overfit tests → more realistic, challenging, *live/continually updated* benchmarks.
- Calls out gaps that block trustworthy deployment evaluations:
  - Cost-efficiency evaluation and standardized reporting.
  - Safety + robustness evaluation (including policy compliance) as first-class outcomes.
  - Need for more fine-grained and scalable methods (trajectory-level diagnostics, automation without fragile judges).

## How is this similar to GALILEO?

- Same meta-goal: reliable measurement of complex, multi-step, interactive behavior (not just single-turn accuracy).
- Reinforces that *trajectory-aware* evaluation and robustness/safety are core missing pieces—aligned with GALILEO’s likely positioning.

## How is this different from GALILEO?

- This is a survey; it does not propose a new benchmark/protocol for the specific failure mode GALILEO targets.
- It aggregates existing benchmarks rather than deeply isolating causal factors (e.g., separating “pressure-driven drift” vs “evidence-driven revision” in multi-turn interactions).

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO provides controlled paired conditions (e.g., neutral vs pressure; evidence vs no-evidence) and multi-turn trajectory diagnostics, it can be positioned as addressing exactly the “fine-grained, scalable, robustness/safety-aware evaluation” gap this survey highlights.

## Where GALILEO is weaker / needs to improve

- If GALILEO lacks explicit cost/efficiency reporting (tokens, tool calls, wall time) or compliance/safety checks, this survey suggests reviewers will increasingly expect those.
- If GALILEO’s scoring is only endpoint-based, it may underdeliver versus the field’s move toward trajectory/milestone metrics.

## Action items for GALILEO (experiments / method / writing)

- [ ] In the related-work / motivation, cite this survey to justify: (i) evaluation fragmentation, (ii) trend toward live/realistic benchmarks, (iii) need for robustness+safety+cost metrics.
- [ ] Add a small “evaluation dimensions” table for GALILEO mapping our benchmark to the survey’s four dimensions (capability/application/generalist/framework) and explicitly state what we cover.
- [ ] Add standardized efficiency reporting: tokens, number of turns, number of tool calls, and (if applicable) wall-clock latency.
- [ ] If applicable, add trajectory-aware reporting (milestones, survival/time-to-failure, recovery) as part of the main results section.

## Quotes / details to potentially cite

- Abstract framing: evaluation across (1) fundamental agent capabilities, (2) application benchmarks, (3) generalist agents, and (4) evaluation frameworks.
- Gap statement from abstract: need better assessment of **cost-efficiency, safety, and robustness**, and more **fine-grained, scalable** evaluation methods.
