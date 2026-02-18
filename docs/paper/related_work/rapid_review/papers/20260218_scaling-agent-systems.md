# Towards a Science of Scaling Agent Systems

- Year: 2025
- Venue: arXiv
- Authors: Yubin Kim, Ken Gu, Chanwoo Park, Chunjong Park, Samuel Schmidgall, A. Ali Heydari, Yao Yan, Zhihan Zhang, Yuchen Zhuang, Mark Malhotra, Paul Pu Liang, Hae Won Park, Yuzhe Yang, Xuhai Xu, Yilun Du, Shwetak Patel, Tim Althoff, Daniel McDuff, Xin Liu
- URL: https://arxiv.org/abs/2512.08296
- BibTeX key (if we add it): kim2025scalingagents
- Tags: agents, scaling, coordination, multi-agent, evaluation, tool-use

## One-sentence takeaway

Agent-system performance follows measurable scaling principles driven by coordination topology, model capability, and task structure, with multi-agent coordination helping parallelizable tasks but often hurting sequential reasoning and tool-heavy workloads.

## What problem does it solve?

- "Multi-agent" LLM systems are widely used, but there is little quantitative guidance for when adding agents / changing coordination structure helps vs. hurts.
- Practitioners lack predictive rules for selecting an agent topology given a task domain (web navigation, planning, tool use) and a fixed compute budget.

## What is the core method / protocol?

- Defines an *agentic evaluation* framing and studies scaling laws over:
  - agent quantity,
  - coordination structure/topology,
  - base model capability,
  - task properties.
- Evaluates 5 canonical architectures:
  - Single-agent,
  - Independent multi-agent,
  - Centralized,
  - Decentralized,
  - Hybrid.
- Controlled sweep across 4 benchmarks and 3 LLM families (180 configurations).
- Fits a predictive model using coordination metrics (paper reports cross-validated R^2=0.524) to predict performance and choose an optimal coordination strategy.

## What are the key metrics?

- Task success / benchmark score per domain (Finance-Agent, BrowseComp-Plus, PlanCraft, Workbench).
- Predictive modeling quality: cross-validated R^2; out-of-sample MAE for an unseen model family (reported MAE=0.071 on GPT-5.2).
- Coordination-related measurements (not fully enumerated on the arXiv abstract page; used as features for the predictive model).

## What are the main results?

- Tool-coordination trade-off: under fixed compute budgets, tool-heavy tasks suffer more from multi-agent overhead.
- Capability saturation: coordination yields diminishing/negative returns once single-agent baseline exceeds ~45%.
- Topology-dependent error amplification:
  - Independent agents amplify errors ~17.2x,
  - Centralized coordination contains error amplification to ~4.4x.
- Centralized coordination: large gains on parallelizable tasks (reported +80.8%).
- Decentralized coordination: best on web navigation (reported +9.2% vs +0.2% for centralized).
- Sequential reasoning tasks: all multi-agent variants degrade performance (reported -39% to -70%).
- Framework selects optimal coordination strategy for 87% of held-out configurations.

## How is this similar to GALILEO?

- Both care about *multi-turn / multi-step* interaction settings where errors can compound across turns.
- Both motivate *task-conditional* evaluation: the right approach depends on task structure (parallelizable vs sequential, tool-heavy vs not).
- The paper's emphasis on stability/error amplification across coordination structures is closely aligned with robustness concerns.

## How is this different from GALILEO?

- Focus is on *agent-system scaling laws and topology selection* (single vs multi-agent), not primarily on user pressure/sycophancy or belief drift.
- Evaluates agent benchmarks (finance, web navigation, planning, workbench), rather than conversational robustness under adversarial multi-turn prompting.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO targets conversational robustness metrics (flip timing, drift, pressure sensitivity), it may provide more direct diagnostic signal for multi-turn *behavioral stability* than agent benchmark aggregate scores.

## Where GALILEO is weaker / needs to improve

- GALILEO may currently lack a principled *task-conditional policy* for when to use single-agent vs multi-agent vs specific coordination topologies.
- If GALILEO includes tool-using settings, this paper suggests explicitly modeling tool overhead and compute budget constraints (multi-agent can harm tool-heavy tasks).

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a "coordination topology" axis to the related-work framing: independent vs centralized vs decentralized vs hybrid; summarize their failure modes (error amplification) and when they help.
- [ ] In experiments, consider a compute-budget-controlled comparison: single-agent vs multi-agent variants under equal token/tool-call budgets.
- [ ] Add a task-structure taxonomy paragraph (parallelizable vs sequential reasoning vs web navigation vs tool-heavy) and cite the reported breakpoints (e.g., saturation beyond ~45% single-agent baseline).

## Quotes / details to potentially cite

- Predictive model for agent-system performance: cross-validated R^2=0.524; out-of-sample MAE=0.071 on an unseen frontier model family.
- Three effects: tool-coordination trade-off; capability saturation (~45% single-agent baseline); topology-dependent error amplification (independent 17.2x vs centralized 4.4x).
- Reported domain-dependent winners: centralized helps parallelizable tasks (+80.8%), decentralized helps web navigation (+9.2% vs +0.2%), multi-agent hurts sequential reasoning (-39% to -70%).
