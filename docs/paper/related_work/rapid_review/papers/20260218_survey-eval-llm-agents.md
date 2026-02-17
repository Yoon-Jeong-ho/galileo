# Survey on Evaluation of LLM-based Agents

- Year: 2025
- Venue: arXiv (survey)
- Authors: Asaf Yehudai; Lilach Eden; Alan Li; Guy Uziel; Yilun Zhao; Roy Bar-Haim; Arman Cohan; Michal Shmueli-Scheuer
- URL: https://arxiv.org/html/2503.16416
- BibTeX key (if we add it): yehudai2025survey_evaluation_llm_agents
- Tags: agents, evaluation, survey, benchmarks, frameworks

## One-sentence takeaway

A broad survey that organizes how to evaluate LLM-based agents (capabilities, application domains, generalist suites, and evaluation frameworks) and highlights gaps around realism, cost/efficiency, safety, and scalable fine-grained evaluation.

## What problem does it solve?

- The evaluation landscape for LLM-based agents is fragmented across many benchmarks/environments (tool-use, web, SWE, science, etc.), making it hard to choose metrics and compare systems.
- Traditional single-call LLM benchmarks miss key agent properties: multi-step trajectories, tool interactions, state/memory, and dynamic environments.

## What is the core method / protocol?

- A structured taxonomy of agent evaluation across four dimensions:
  1) Fundamental agent capabilities: planning/multi-step reasoning, function calling/tool use, self-reflection, memory.
  2) Application-specific agent benchmarks: web, software engineering, scientific, conversational.
  3) Generalist agent benchmarks/leaderboards: multi-skill task suites for end-to-end agents.
  4) Evaluation frameworks: developer tooling and “gym-like” environments supporting iterative evaluation.
- Discussion of trends (e.g., more realistic + continuously updated/live benchmarks) and open problems.

## What are the key metrics?

- Predominantly task success / completion rate (end-to-end), often with variants for:
  - efficiency: steps, time, cost (tokens, tool calls), latency
  - robustness: performance under interface changes / distribution shift
  - intermediate milestone completion / trajectory-level scoring
  - correctness/accuracy depending on domain (e.g., tests passing in SWE)
- Notes that safety/policy compliance and cost-efficiency metrics are underdeveloped relative to success-rate metrics.

## What are the main results?

- Not an empirical contribution; main “results” are synthesized observations:
  - Benchmark evolution is moving from static/offline tasks to more realistic, dynamic, and continuously updated evaluations.
  - Current evaluations still emphasize success rate; gaps remain in scalable fine-grained evaluation, robustness, safety/compliance, and cost/efficiency.
  - Capability evaluations overlap with classic reasoning tasks but need agent-specific adaptations (interactive, stateful).

## How is this similar to GALILEO?

- Directly relevant if GALILEO positions itself as an agent evaluation approach/system: the paper surveys evaluation frameworks and generalist agent leaderboards (it explicitly lists “Galileo’s Agent Leaderboard”).
- Emphasizes realistic, dynamic evaluation and continuous updates—common themes in agent benchmarking/leaderboards.

## How is this different from GALILEO?

- This is a meta-level survey/taxonomy rather than proposing a specific benchmark, environment, metric suite, or evaluation product.
- Focuses on cataloging existing benchmarks/frameworks rather than introducing new protocols or datasets.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO provides standardized, continuously updated, reproducible evaluation with cost/safety reporting, it directly addresses gaps the survey highlights.

## Where GALILEO is weaker / needs to improve

- Survey flags open problems that are easy to miss in leaderboard-centric evaluation:
  - fine-grained trajectory evaluation (intermediate milestones, error recovery)
  - robust cost/efficiency accounting across models/tools
  - safety/policy compliance measurement beyond task success

## Action items for GALILEO (experiments / method / writing)

- [ ] Map GALILEO’s benchmark(s)/leaderboard(s) onto the survey’s four-way taxonomy (capabilities, application-specific, generalist, frameworks) and state coverage explicitly.
- [ ] Add/strengthen reporting for cost-efficiency (tokens, tool calls, wall-clock) and robustness (e.g., interface/version drift) as first-class metrics.
- [ ] Consider milestone/trajectory-level scoring (partial credit) for long-horizon tasks to reduce brittleness of pure success-rate.
- [ ] Add explicit safety/compliance evaluation hooks if applicable (policy constraints, refusal correctness, secure tool use).

## Quotes / details to potentially cite

- “We systematically analyze evaluation benchmarks and frameworks across four critical dimensions: (1) fundamental agent capabilities … (2) application-specific benchmarks … (3) benchmarks for generalist agents; and (4) frameworks for evaluating agents.”
- “Our analysis reveals emerging trends, including a shift toward more realistic, challenging evaluations with continuously updated benchmarks.”
- “We also identify critical gaps … particularly in assessing cost-efficiency, safety, and robustness, and in developing fine-grained, and scalable evaluation methods.”
