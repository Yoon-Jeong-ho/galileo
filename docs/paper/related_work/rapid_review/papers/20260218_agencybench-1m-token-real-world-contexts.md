# AgencyBench: Benchmarking the Frontiers of Autonomous Agents in 1M-Token Real-World Contexts

- Year: 2026
- Venue: arXiv
- Authors: Keyu Li; Junhao Shi; Yang Xiao; Mohan Jiang; Jie Sun; Yunze Wu; Shijie Xia; Xiaojie Cai; Tianze Xu; Weiye Si; Wenjie Li; Dequan Wang; Pengfei Liu
- URL: https://arxiv.org/abs/2601.11044
- BibTeX key (if we add it): AgencyBench2026Li
- Tags: agents, long-horizon, tool-use, multi-turn, benchmark, evaluation, user-simulation, sandbox

## One-sentence takeaway

AgencyBench is a long-horizon, tool-heavy benchmark (32 scenarios / 138 tasks) that uses user-simulation plus Docker-based rubric checks to scale evaluation of autonomous agents over ~1M-token, ~90-tool-call trajectories.

## What problem does it solve?

- Existing agent benchmarks often target a single capability or short-horizon tasks, which misses realistic, end-to-end “daily AI usage” workflows.
- Realistic evaluation tends to require human-in-the-loop feedback, limiting scalability for collecting/evaluating long rollouts.

## What is the core method / protocol?

- Benchmark construction:
  - 32 real-world scenarios, 138 tasks with explicit queries, deliverables, and rubrics.
  - Evaluates 6 “core agentic capabilities” (paper-level categories; not enumerated on the arXiv abstract page).
  - Tasks are intentionally long-horizon: reported average ~90 tool calls and ~1M tokens, “hours of execution time”.
- Automated evaluation stack:
  - A **user simulation agent** provides iterative feedback during execution (to replace human-in-the-loop).
  - A **Docker sandbox** runs tools and supports **visual + functional rubric-based assessment** of deliverables.
- Experiments:
  - Compares closed vs open models; analyzes resource efficiency, self-correction under feedback, tool-use preferences.
  - Studies effect of different agentic scaffolds / ecosystems (e.g., model + SDK pairing).

## What are the key metrics?

- Task success / rubric score (implied by “rubrics” and reported aggregate performance numbers).
- Resource efficiency indicators (tokens, tool calls, time/cost proxies) discussed as an analysis dimension.
- Feedback-driven self-correction capability (analysis dimension).

## What are the main results?

- Closed-source models outperform open-source models on this benchmark (reported 48.4% vs 32.1%).
- Performance varies notably by:
  - Efficiency (how much compute/tool usage is needed)
  - Ability to improve via iterative feedback
  - Tool-use preferences
- Scaffold/ecosystem interactions matter:
  - Proprietary models do best in their “native” agent stacks.
  - Open-source models show distinct peaks depending on execution framework, suggesting tuning opportunities.

## How is this similar to GALILEO?

- If GALILEO involves long-horizon, multi-step/tool-using agent behavior, AgencyBench is directly relevant as:
  - A reference benchmark for **real-world scenario design** and rubric structuring.
  - An example of scaling evaluation via **simulated user feedback** + **sandboxed functional checks**.

## How is this different from GALILEO?

- AgencyBench is primarily an **evaluation benchmark + tooling** contribution, not a new agent policy/model.
- The benchmark emphasizes extremely long contexts (up to ~1M tokens) and heavy tool-call budgets as first-class properties.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO targets a narrower, well-controlled domain/task family, it may offer cleaner causal attribution than a broad “daily usage” benchmark.

## Where GALILEO is weaker / needs to improve

- If GALILEO evaluation is still human-driven or short-horizon, AgencyBench highlights a gap:
  - Need scalable feedback loops and automated rubric checks.
  - Need stress-testing for long-context degradation and tool-call compounding errors.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding a **user-simulator** for iterative clarification/feedback to reduce human evaluation load.
- [ ] Add **Docker/sandbox rubric checks** (functional + basic UI/visual checks) for end-to-end tasks.
- [ ] In related work, cite AgencyBench as evidence that benchmark outcomes depend on **agent scaffolds/ecosystems**, not only base model.
- [ ] If feasible, add a “long-horizon” slice to GALILEO eval: measure success vs tool-call budget and context length.

## Quotes / details to potentially cite

- “32 real-world scenarios, comprising 138 tasks with specific queries, deliverables, and rubrics.”
- “These scenarios require an average of 90 tool calls, 1 million tokens, and hours of execution time to resolve.”
- “To enable automated evaluation, we employ a user simulation agent … and a Docker sandbox to conduct visual and functional rubric-based assessment.”
- “closed-source models significantly outperform open-source models (48.4% vs 32.1%).”
- Benchmark/toolkit release: https://github.com/GAIR-NLP/AgencyBench
