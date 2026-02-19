# Team of Thoughts: Efficient Test-time Scaling of Agentic Systems through Orchestrated Tool Calling

- Year: 2026
- Venue: arXiv
- Authors: Jeffrey T. H. Wong, Zixi Zhang, Junyi Liu, Yiren Zhao
- URL: https://arxiv.org/abs/2602.16485
- BibTeX key (if we add it): wong2026team
- Tags: agents, orchestration, tool-calling, test-time-scaling, heterogeneous-models

## One-sentence takeaway

A hierarchical multi-agent setup (“orchestrator + tool agents”) that *calibrates* which model should orchestrate and uses *agent self-assessed specialization* to selectively invoke heterogeneous models for better accuracy-per-token at test time.

## What problem does it solve?

- Standard test-time scaling (CoT/ToT/beam/PRM) is often token-inefficient and bounded by a single model’s parameters.
- Many multi-agent systems are effectively *homogeneous* (same base model role-played), so they don’t get meaningful diversity of priors.
- Even when heterogeneous models are available, existing MAS commonly use fixed workflows/roles rather than dynamically selecting specialists per query.

## What is the core method / protocol?

- **Team-of-Thoughts (ToTh)**: treat each LLM as a callable “tool agent” and have a separate **orchestrator** LLM that:
  1) selects which tool agents to invoke for a query,
  2) evaluates/synthesizes tool outputs,
  3) produces the final answer.

Two key mechanisms:

- **Orchestrator calibration:** choose the orchestrator model by measuring which candidate orchestrator best aggregates tool-agent responses on a *category-specific* calibration set under a cost budget.
- **Tool-agent specialization profiles via self-assessment:** for each tool agent and category, compute a proficiency score on validation data; at inference time the orchestrator activates a subset of agents based on these scores (and can allocate more tokens to the best-matched agents).

Efficiency claims:

- **Parallelism:** tool agents can run concurrently (one “round” of calls).
- **Strategic token allocation:** skip low-proficiency agents, request short answers from marginal agents, spend budget on the most relevant specialists.

## What are the key metrics?

- Accuracy on reasoning and code generation benchmarks.
- (Implicit) efficiency via selective activation / token budget (paper frames as better performance-to-token vs baselines).

## What are the main results?

- Reported gains over homogeneous role-play MAS baselines on five benchmarks.
- Highlighted numbers (from abstract):
  - **AIME24:** 96.67% (vs 80% for a homogeneous role-play baseline cited as AgentVerse).
  - **LiveCodeBench:** 72.53% (vs 65.93% baseline).

## How is this similar to GALILEO?

- Shares the core idea that **agentic systems benefit from explicit orchestration and structured coordination**, rather than “one model does everything”.
- Emphasizes **selective invocation** (not all agents/tools every time), which aligns with building controllable, efficient agent pipelines.

## How is this different from GALILEO?

- This work is primarily about **heterogeneous LLM ensembling via an orchestrator-tool interface**, whereas GALILEO may focus more on (likely) planning/control, environment interaction, or a specific agent architecture beyond model selection.
- Their “self-assessment” is essentially **benchmark-derived proficiency per category**, not necessarily online uncertainty estimation, introspective confidence calibration, or learned routing.
- Evaluation focus is on **reasoning + code benchmarks** (AIME/LiveCodeBench, etc.) rather than interactive tasks/environments (if GALILEO targets those).

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO provides a principled decomposition of tasks, formal guarantees, or environment-grounded evaluation, that can be a stronger story than “choose the best model/router on held-out sets”.
- GALILEO may avoid relying on *a priori* category labels (this paper assumes/uses task categories for calibration/routing).

## Where GALILEO is weaker / needs to improve

- If GALILEO currently uses a mostly homogeneous agent set, this paper is a reminder that **true heterogeneity** (different model priors / post-training) can matter.
- If GALILEO lacks a clean routing story, the **calibrate-orchestrator + proficiency-profile** framing could be a useful baseline to beat.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a related-work paragraph: “heterogeneous MAS via orchestrator/tool calling; calibrating orchestrator model; profiling agent specialization; selective activation for token efficiency” (cite this arXiv).
- [ ] Consider an ablation or baseline: **static homogeneous MAS vs heterogeneous toolset with routing**.
- [ ] If GALILEO uses tool-calling, compare to **one-round parallel tool calls** with an orchestrator as a strong test-time scaling baseline.

## Quotes / details to potentially cite

- “Existing Multi-Agent Systems (MAS) typically rely on static, homogeneous model configurations…”
- “Our framework introduces two key mechanisms… (1) an orchestrator calibration scheme… and (2) a self-assessment protocol…”
- “During inference, the orchestrator dynamically activates the most suitable tool agents based on these proficiency profiles.”
