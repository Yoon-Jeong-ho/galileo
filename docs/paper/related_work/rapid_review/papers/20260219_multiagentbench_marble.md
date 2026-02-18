# MultiAgentBench: Evaluating the Collaboration and Competition of LLM agents

- Year: 2025
- Venue: arXiv
- Authors: Kunlun Zhu; Hongyi Du; Zhaochen Hong; Xiaocheng Yang; Shuyi Guo; Zhe Wang; Zhenhailong Wang; Cheng Qian; Xiangru Tang; Heng Ji; Jiaxuan You
- URL: https://arxiv.org/abs/2503.01935
- BibTeX key (if we add it): MultiAgentBench2025Zhu
- Tags: multi-agent, benchmarking, evaluation, coordination, competition, LLM-agents

## One-sentence takeaway

MultiAgentBench (with the MARBLE framework) is a multi-domain benchmark + harness for evaluating LLM multi-agent systems that scores not just task success but also milestone-based coordination/competition quality across different communication topologies and planning strategies.

## What problem does it solve?

- Existing agent benchmarks largely target single-agent capability or narrow domains, and do not capture multi-agent dynamics like coordination structure, communication quality, and explicit competition.
- Practitioners lack standardized, task-diverse evaluation that can compare coordination protocols (star/tree/chain/graph) and planning strategies (group discussion, “cognitive” self-evolving planning) in interactive scenarios.

## What is the core method / protocol?

- Introduces **MultiAgentBench**, and an accompanying evaluation backbone called **MARBLE** (Multi-agent cooRdination Backbone with LLM Engine).
- MARBLE components (high-level):
  - **Coordination engine** that instantiates agents and their relationships.
  - **Agent graph** representation for who can communicate with whom (edges encode relations like collaboration/supervision/negotiation).
  - **Planner vs actor** split (planners allocate/plan; actors execute in environments/tools).
- Evaluates **coordination protocols / communication topologies**:
  - Centralized: **star**, **tree**.
  - Decentralized: **graph-mesh**, **chain**.
- Evaluates **planning prompt strategies** for the planner:
  - Vanilla, CoT-style planning, group discussion, and a “cognitive self-evolving planning” approach (stores expectations/outcomes in memory and compares across iterations, Reflexion-like).
- Benchmark design: multiple interactive scenarios spanning task-oriented collaboration and social-simulation/competition settings (paper claims 6 scenarios; examples mentioned include research collaboration, Minecraft building, database error analysis, coding; also social games like Werewolf/bargaining).

## What are the key metrics?

- **Task score / completion** (scenario-specific outcomes).
- **Milestone-based KPI**: tracks progress via dynamically detected milestones (intended to quantify partial progress, not just binary success).
- Additional scores intended to reflect **planning/communication quality** and an explicit **competition score** for conflicting-goal tasks.

## What are the main results?

- Reported highlights (from abstract):
  - **gpt-4o-mini** achieves the highest average task score among tested models.
  - **Graph** coordination performs best among coordination protocols in the **research** scenario.
  - **Cognitive planning** increases milestone achievement rates by **~3%**.

## How is this similar to GALILEO?

- Both care about **agentic evaluation beyond single-turn QA**, including structured multi-step tasks.
- Emphasis on **process-aware metrics** (milestones / intermediate objectives) rather than only final success.
- Explicitly varies **coordination structures** (topologies) and planning strategies, which is relevant if GALILEO compares orchestration variants.

## How is this different from GALILEO?

- Focus is explicitly on **multi-agent coordination + competition dynamics** across interactive environments, whereas GALILEO may be more centered on a specific agent architecture or a narrower task family.
- Their harness (MARBLE) is a benchmark/execution backbone with **agent-graph relations** and configurable communication topology as first-class objects.
- Their evaluation includes **competition scenarios** and a “competition score” component, which may not be central in GALILEO.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO has tighter experimental controls (fixed tools/envs, clearer ablations, stronger baselines), it can present a cleaner causal story than a broad multi-scenario benchmark suite.
- If GALILEO has stronger reproducibility packaging (exact env snapshots, deterministic eval), that can be a differentiator; MultiAgentBench spans several interactive settings that may be harder to reproduce.

## Where GALILEO is weaker / needs to improve

- If GALILEO does not yet have:
  - explicit **multi-agent topology** comparisons,
  - **milestone/partial-credit** scoring,
  - or **competitive** multi-agent settings,
  then MultiAgentBench is a clear pointer to what reviewers may expect in “multi-agent evaluation” sections.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding a **milestone-based KPI** (partial credit) for at least one GALILEO scenario to capture progress and coordination quality.
- [ ] Add an ablation over **coordination topology** (star/tree/chain/graph) if GALILEO includes multiple agents or modular sub-agents.
- [ ] If relevant, add at least one **conflicting-goal / competition** task, or explain why GALILEO excludes that axis.
- [ ] In related work, position GALILEO relative to **MARBLE/MultiAgentBench** as a benchmark + coordination backbone (emphasize what GALILEO measures differently).

## Quotes / details to potentially cite

- “existing benchmarks either focus on single-agent tasks or are confined to narrow domains, failing to capture the dynamics of multi-agent coordination and competition” (abstract).
- Bench evaluates “coordination protocols (including star, chain, tree, and graph topologies)” and strategies “group discussion and cognitive planning” (abstract).
- “cognitive planning improves milestone achievement rates by 3%” (abstract).
