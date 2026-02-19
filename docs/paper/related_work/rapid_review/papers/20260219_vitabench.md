# VitaBench: Benchmarking LLM Agents with Versatile Interactive Tasks in Real-world Applications

- Year: 2025
- Venue: arXiv
- Authors: Wei He, Yueqing Sun, Hongyan Hao, Xueyuan Hao, Zhikang Xia, Qi Gu, Chengcheng Han, Dengchang Zhao, Hui Su, Kefeng Zhang, Man Gao, Xi Su, Xiaodong Cai, Xunliang Cai, Yu Yang, Yunke Zhao
- URL: https://arxiv.org/abs/2509.26490
- BibTeX key (if we add it): vitabench2025
- Tags: agents, benchmark, multi-turn, tool-use, interactive, cross-domain, evaluation

## One-sentence takeaway

VitaBench is a multi-turn, tool-rich “life services” agent benchmark (delivery / in-store / travel) with cross-scenario tasks and a rubric-based sliding-window evaluator, where even strong LLMs reach ~30% success on cross-scenario settings.

## What problem does it solve?

- Existing agent benchmarks under-model real deployments where agents must (i) integrate lots of environment information, (ii) navigate interdependent tool ecosystems, and (iii) handle uncertain, evolving multi-turn user interactions.
- Prior work often evaluates tool calling in isolation, constrains trajectories with rigid policies, or under-represents user uncertainty/intent shifts.

## What is the core method / protocol?

- Benchmark construction:
  - “Life-serving” simulation environments spanning three domains: food delivery, in-store consumption, and online travel services.
  - Tooling: 66 tools with explicit inter-tool dependencies (they emphasize representing tools as a dependency graph; policies are “encoded” by the environment/tool structure rather than a separate policy document).
  - Tasks: 300 single-scenario tasks + 100 cross-scenario tasks.
  - Each task is derived from multiple user requests, with spatiotemporal context and user profiles/attributes.
- Evaluation:
  - A rubric-based sliding window evaluator meant to grade long-horizon trajectories when multiple solution paths can be valid and interactions are stochastic.

## What are the key metrics?

- Primary: task success rate.
- Diagnostic: they propose/track three “task complexity” dimensions and correlate with difficulty:
  - Reasoning complexity (amount/structure of information to integrate)
  - Tool complexity (graph size / dependency density)
  - Interaction complexity (user behavior attributes; multi-turn patterns)
- Failure breakdown (from intro): reasoning errors vs tool usage errors vs interaction management failures.

## What are the main results?

- Best model performance is low:
  - ~48.3% success on single-scenario tasks.
  - ~30.0% success on cross-scenario tasks.
- Reported dominant failure category: reasoning errors (61.8%), then tool usage errors (21.1%), then interaction management failures (7.9%).

## How is this similar to GALILEO?

- Same general target: evaluating agentic behavior in realistic, multi-step, multi-turn settings with tool use and ambiguous/evolving user intent.
- Emphasizes long-horizon trajectories (not just single function-call correctness) and the importance of robust evaluation when there are multiple valid plans.

## How is this different from GALILEO?

- Domain focus is “life services” (delivery / in-store / travel), rather than scientific/analytic or other GALILEO target domains.
- Their benchmark stresses inter-tool dependency structure as a first-class driver of difficulty (graph modeling), and explicitly includes cross-scenario tasks that require switching contexts/domains.
- Their evaluator is a specific rubric + sliding-window grading approach; depending on GALILEO’s evaluator, this could be a different philosophy than strict end-state checking or single-score judges.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO provides clearer ground-truth signals, tighter controllability, or better decomposition/attribution across capabilities, it may yield more interpretable comparisons than a broad “life services” simulator.
- If GALILEO already supports stronger ablations (tool-graph structure, user stochasticity), it can offer cleaner causal claims.

## Where GALILEO is weaker / needs to improve

- Consider whether GALILEO adequately covers:
  - Cross-scenario / cross-domain tool selection (expanded action spaces).
  - Explicit modeling/quantification of tool dependency structure as part of task design.
  - User-profile-driven interaction complexity and intent shift across many turns.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add (or at least report) a “tool dependency graph complexity” metric for GALILEO tasks (node count, edge density, path lengths).
- [ ] If applicable, introduce a small “cross-scenario” split where agents must transfer state between subtasks/domains.
- [ ] Consider a sliding-window / progress-based rubric evaluator for long-horizon tasks, or at minimum report partial-credit/progress signals.
- [ ] In related work, position GALILEO against VitaBench along the three complexity axes (reasoning/tool/interaction) rather than only benchmark size.

## Quotes / details to potentially cite

- “VitaBench presents agents with the most complex life-serving simulation environment to date, comprising 66 tools.”
- “	... yielding 100 cross-scenario tasks (main results) and 300 single-scenario tasks.”
- “Even the best-performing model achieves only 48.3% success rate across the single-scenario tasks ... performance plummeting to 30.0% in cross-scenario settings.”
- Failure analysis (intro): reasoning errors 61.8%, tool usage errors 21.1%, interaction management failures 7.9%.