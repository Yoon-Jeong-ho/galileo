# DataGovBench: Benchmarking LLM Agents for Real-World Data Governance Workflows

- Year: 2025
- Venue: arXiv
- Authors: Zhou Liu; Zhaoyang Han; Guochen Yan; Hao Liang; Bohan Zeng; Xing Chen; Yuanfeng Song; Wentao Zhang
- URL: https://arxiv.org/abs/2512.04416
- BibTeX key (if we add it): datagovbench2025liu
- Tags: agents, benchmarking, data-governance, workflows, planner-executor-evaluator, debugging

## One-sentence takeaway

A 150-task benchmark for natural-language-driven **data governance** pipelines (with “reversed-objective” noise synthesis and runnable evaluation scripts), plus a Planner–Executor–Evaluator agent that improves end-to-end task success and reduces debugging iterations.

## What problem does it solve?

- Existing “data science” LLM benchmarks over-index on snippet coding or high-level analytics, and do not directly test the core difficulty of **data governance**: producing an output dataset that is *correct, clean, and compliant* end-to-end.
- Evaluating NL→ETL / NL→data-cleaning workflows needs (i) realistic noisy inputs, (ii) executable evaluation scripts, and (iii) metrics that reflect pipeline reliability rather than just code plausibility.

## What is the core method / protocol?

- **DataGovBench (benchmark)**
  - 150 tasks: **100 operator-level** + **50 DAG-level**.
  - 6 scenarios (operator-level): Filtering; Refinement; Imputation; Deduplication & Consistency; Data Integration; Classification & Labeling.
  - Built from real-world tables; tasks described in natural language; includes both structured and some unstructured fields.
  - Key idea: **“reversed-objective” noise synthesis**: invert the task objective to generate task-specific disruption/noise code, aiming for realistic, measurable corruptions.
  - Provides **task-specific evaluation scripts** that compare processed data against ground truth and output normalized scores.

- **DataGovAgent (agent framework)**
  - Planner–Executor–Evaluator (“agentic assembly line”).
  - Planner: converts NL intent into a DAG of operations and adds runtime checks.
  - Executor: retrieval-augmented generation over a curated library to produce code.
  - Evaluator: sandboxed execution + iterative feedback-driven debugging loop.

## What are the key metrics?

- **CRR**: Code Runnable Rate.
- **TSR**: Task Success Rate (main business-objective metric).
- **ATS**: Average Task Score (normalized score; paper reports ATS improvements on complex tasks).
- **ADI**: Average Debug Iterations (efficiency of debugging loop; reported vs baselines).

## What are the main results?

- Paper reports that general-purpose baselines struggle with complex multi-step workflows and robust error correction.
- DataGovAgent improvements (as reported in abstract/intro):
  - Complex-task **ATS** improves from **39.7 → 54.9**.
  - Debugging iterations reduced by **~77.9%** vs general-purpose baselines.
  - On DataGovBench-150 with GPT-5: **TSR** to **64%** (operator-level) and **60%** (DAG-level), with notable gains over prior frameworks; **ADI** reduced substantially vs ChatDev.

## How is this similar to GALILEO?

- Shares the broad framing of **LLM agents** for real workflows: decomposition/planning, code generation, execution-time feedback, and iterative correction.
- Provides a concrete example of **benchmarking long(er)-horizon agent behavior** beyond single-shot code generation.

## How is this different from GALILEO?

- Domain focus is **data governance / ETL correctness** (dataset-level quality) rather than GALILEO’s primary target domain.
- Their evaluation emphasizes *pipeline output correctness* (dataset comparisons + runnable scripts), not conversational robustness or multi-turn social pressure.
- DataGovAgent is explicitly a **code-execution** and **debugging** loop architecture; if GALILEO is not code-executing, direct comparisons are limited.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO’s contribution centers on interaction-level robustness (multi-turn dynamics, safety, or stability), this paper is mostly orthogonal: it does not appear to provide deep behavioral/interaction protocols beyond code debugging.

## Where GALILEO is weaker / needs to improve

- Benchmarking: DataGovBench illustrates a strong pattern for building **task-specific runnable evaluators** and normalized scoring—useful if GALILEO needs more “executable” evaluation artifacts.
- Reliability framing: CRR/TSR/ATS-style reporting is reviewer-friendly for end-to-end success and could complement any GALILEO metrics that are harder to interpret.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider whether a **“reversed-objective”** approach could generate realistic perturbations/noise in GALILEO’s domain (i.e., programmatically construct *hard-but-measurable* corruptions).
- [ ] If GALILEO has agentic workflows, consider adding an efficiency metric analogous to **ADI** (debug/repair iterations, backtracks, or tool-call cycles).
- [ ] In related work, potentially cite DataGovBench as an example of moving from snippet-level evaluation to **end-to-end pipeline reliability** via runnable metrics.

## Quotes / details to potentially cite

- DataGovBench: “150 real-world tasks (100 operator-level; 50 DAG-level)” across six governance scenarios.
- Benchmark methodology: “reversed-objective” noise synthesis and “task-specific evaluation scripts” with CRR/TSR/ATS.
- Agent: Planner–Executor–Evaluator architecture with retrieval-augmented generation and sandboxed feedback-driven debugging.
- Reported headline numbers: complex-task ATS 39.7→54.9; debugging iterations −77.9%; TSR up to ~64% (operator) / 60% (DAG) with GPT-5.
