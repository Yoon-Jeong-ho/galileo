# OdysseyBench: Evaluating LLM Agents on Long-Horizon Complex Office Application Workflows

- Year: 2025
- Venue: arXiv
- Authors: Weixuan Wang; Dongge Han; Daniel Madrigal Diaz; Jin Xu; Victor Rühle; Saravan Rajmohan
- URL: https://arxiv.org/abs/2508.09124
- BibTeX key (if we add it): odysseybench_wang_2025
- Tags: agents, long-horizon, workflows, evaluation, office-apps

## One-sentence takeaway

OdysseyBench is a long-horizon, multi-application (Word/Excel/PDF/Email/Calendar) benchmark plus a multi-agent generation pipeline (HomerAgents) meant to stress-test LLM agents beyond atomic tasks.

## What problem does it solve?

- Existing agent benchmarks often emphasize *atomic*, self-contained tasks and under-measure long-horizon workflows with contextual dependencies and multiple user–agent interactions.
- Lack of scalable processes for creating long-horizon, tool-using workflow benchmarks without heavy human annotation.

## What is the core method / protocol?

- **OdysseyBench** benchmark with two splits:
  - **OdysseyBench+**: 300 long-horizon tasks derived from real-world use cases (built on/related to OfficeBench).
  - **OdysseyBench-Neo**: 302 newly synthesized tasks intended to be more complex/diverse.
- Each task requires the agent to:
  - Extract essential information from *long-horizon dialogue histories*.
  - Perform *multi-step reasoning* spanning multiple office applications.
- **HomerAgents** (multi-agent framework) to create the benchmark at scale:
  - A process that transforms or generates tasks via environment exploration, task generation, and dialogue synthesis.
- The paper also highlights an analysis point: **dialogue storage formats** matter; it claims semantic compression / coherent aggregation is important for multi-step reasoning performance.

## What are the key metrics?

- Primary notion is agent **task performance on benchmark tasks** (e.g., task success / completion in the office-workflow environments).
- Sensitivity to **how long-horizon interaction history is stored/presented** (storage/format ablations).

*(Detailed metric definitions and scoring were not reviewed beyond abstract/intro.)*

## What are the main results?

- The benchmark is reported to **challenge state-of-the-art LLM agents** more than prior atomic-task benchmarks, providing a more realistic assessment for productivity workflows.
- The paper reports that **semantic compression and coherent aggregation** of dialogue/history are important for effective multi-step reasoning in this setting.

## How is this similar to GALILEO?

- Both care about **multi-turn / long-horizon evaluation**, where interaction history and context accumulation can substantially affect measured capability.
- Both point toward *evaluation protocol choices* (what context is shown; how it is summarized/aggregated) as a major determinant of observed behavior.

## How is this different from GALILEO?

- OdysseyBench is primarily about **tool-using office productivity workflows** (cross-application task execution), not directly about **truth/consistency drift** or *being misled* over turns.
- GALILEO’s core focus (as used in our related-work positioning) is closer to **stability/robustness over interaction** (e.g., drift vs evidence-driven revision) rather than general long-horizon task success.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO emphasizes **controlled perturbations** and clearer separation of *drift* vs *legitimate update*, it can provide a cleaner causal story than a broad task benchmark.

## Where GALILEO is weaker / needs to improve

- GALILEO may not cover the breadth of **realistic, multi-application tool workflows** and their evaluation infrastructure.
- If we do not ablate **history representation** (raw log vs compressed/structured), we may miss an important confound that OdysseyBench explicitly surfaces.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a short related-work paragraph: “atomic vs long-horizon benchmarks” + cite OdysseyBench as evidence that **long-horizon context handling** changes conclusions.
- [ ] Consider an ablation: **history formatting/aggregation** (raw transcript vs structured summary vs semantic compression) and measure impact on GALILEO’s drift/recovery metrics.
- [ ] If relevant, borrow taxonomy language: “long-horizon contextual dependencies” and “multi-interaction coordination”.

## Quotes / details to potentially cite

- “existing benchmarks predominantly focus on atomic tasks … failing to capture the long-term contextual dependencies and multi-interaction coordination required in realistic scenarios.” (abstract)
- OdysseyBench includes “Word, Excel, PDF, Email, and Calendar” workflows and splits into “OdysseyBench+ … 300 tasks …” and “OdysseyBench-Neo … 302 … tasks.” (abstract)
- Paper claims “semantic compression and coherent aggregation are essential” when analyzing dialogue storage formats (intro contribution list).
