# OdysseyBench: Evaluating LLM Agents on Long-Horizon Complex Office Application Workflows

- Year: 2025
- Venue: arXiv
- Authors: Weixuan Wang et al. (see arXiv)
- URL: https://arxiv.org/abs/2508.09124
- BibTeX key (if we add it): odysseybench2025
- Tags: agents, long-horizon, office, workflows, benchmark

## One-sentence takeaway

OdysseyBench is a long-horizon benchmark of multi-application “office workflow” tasks (Word/Excel/PDF/Email/Calendar) plus a framework (HomerAgents) for generating such workflows at scale.

## What problem does it solve?

- Existing agent benchmarks over-index on short, atomic, single-tool tasks, under-measuring long-horizon context dependence and cross-application coordination typical of productivity workflows.
- Need for scalable creation of realistic, multi-step tasks that require tracking information across long interaction histories.

## What is the core method / protocol?

- **OdysseyBench** benchmark with two splits:
  - **OdysseyBench+**: 300 tasks derived from real-world use cases.
  - **OdysseyBench-Neo**: 302 newly synthesized complex tasks.
- Task setting: long-horizon interaction histories where an agent must:
  - identify essential information from prior interactions
  - execute multi-step actions spanning office apps (Word/Excel/PDF/Email/Calendar)
- **HomerAgents**: a multi-agent framework to automate benchmark creation via:
  - environment exploration
  - task generation
  - dialogue/history synthesis

## What are the key metrics?

- Not specified in the abstract (paper likely reports task success / completion accuracy and per-app breakdowns).

## What are the main results?

- Not specified in the abstract; the authors claim OdysseyBench “effectively challenges” SOTA LLM agents and is more diagnostic for long-horizon, real-world productivity workflows than atomic-task benchmarks.

## How is this similar to GALILEO?

- Shares the motivation that **multi-turn / long-horizon** evaluation surfaces failures that do not appear in single-turn/atomic benchmarks.
- Reinforces the need to evaluate agents under realistic **interaction-history dependencies**.

## How is this different from GALILEO?

- Focus is **tool-using productivity workflows** (GUI/office app execution), not primarily belief/stance drift under social pressure or robustness-to-manipulation protocols.
- Likely emphasizes end-task success rather than detailed **time-to-failure / recovery / flip-dynamics** metrics.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO centers controlled conversational/pressure operators, it can provide cleaner causal attribution (pressure vs evidence) and richer trajectory metrics than “did the workflow succeed?”.

## Where GALILEO is weaker / needs to improve

- If GALILEO lacks realistic cross-tool tasks, OdysseyBench highlights a gap: long-horizon failures often arise from **context management across heterogeneous tools**.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding (or at least discussing) a “productivity workflow” slice as an *external validity* reference point for long-horizon agent evaluation.
- [ ] In related work, position GALILEO as complementary: OdysseyBench measures workflow competence; GALILEO measures robustness dynamics (drift/recovery) that can occur within or across workflows.

## Quotes / details to potentially cite

- “Existing benchmarks predominantly focus on atomic tasks that are self-contained and independent, failing to capture the long-term contextual dependencies and multi-interaction coordination required in realistic scenarios.”
- “We introduce OdysseyBench, a comprehensive benchmark for evaluating LLM agents on long-horizon workflows across diverse office applications including Word, Excel, PDF, Email, and Calendar.”
- “Our benchmark comprises two complementary splits: OdysseyBench+ with 300 tasks … and OdysseyBench-Neo with 302 … synthesized complex tasks.”
- “We propose HomerAgents, a multi-agent framework that automates the generation of long-horizon workflow benchmarks through systematic environment exploration, task generation, and dialogue synthesis.”
