# CodeFlowBench: A Multi-turn, Iterative Benchmark for Complex Code Generation

- Year: 2025 (v3 posted 2026-01)
- Venue: arXiv
- Authors: Sizhe Wang, Zhengren Wang, Dongsheng Ma, Yongan Yu, Rui Ling, Zhiyu Li, Feiyu Xiong, Wentao Zhang
- URL: https://arxiv.org/abs/2504.21751
- BibTeX key (if we add it): codeflowbench_wang_2025
- Tags: multi-turn, evaluation, benchmark, code-generation, iterative, dependency-structure

## One-sentence takeaway

CodeFlowBench proposes a large, updateable benchmark to measure how code LLMs degrade in *multi-turn, iterative* development when they must reuse previously-written functions, with analysis tied to dependency-tree complexity.

## What problem does it solve?

- Existing code-generation benchmarks are mostly single-turn (or limited to small edits) and don’t capture iterative workflows where developers implement functionality step-by-step while *reusing* earlier code.
- Prior multi-turn code benchmarks often focus on single-function tasks and may lack strong verification (tests) and/or realistic dependency structure.
- Static datasets risk contamination; authors emphasize frequent updates.

## What is the core method / protocol?

- Define **codeflow**: multi-turn implementation where each turn adds a module/function while leveraging previously implemented functions.
- Build **CodeFlowBench** with two parts:
  - **CodeFlowBench-Comp**: 5,000+ problems sourced from competitive programming (Codeforces), updated via an automated pipeline.
  - **CodeFlowBench-Repo**: tasks sourced from GitHub repositories to reflect real-world software structure.
- Automated curation pipeline based on **function dependency analysis**: decompose a monolithic solution into a sequence of subproblems aligned with a dependency tree (bottom-up reuse).
- Evaluation framework includes a **dual assessment protocol** plus **structural metrics** computed from dependency trees (to analyze how structure/complexity affects outcomes).

## What are the key metrics?

- Primary outcome: task success under multi-turn codeflow setting (verified via provided tests).
- Structural metrics from dependency trees (e.g., dependency complexity / depth / branching; exact definitions are paper-specific) used to correlate structure with performance.
- Reported analysis focus: performance vs dependency complexity and multi-turn degradation (how accuracy drops over turns / compared to single-turn).

## What are the main results?

- Across evaluated models, performance **drops substantially** in multi-turn codeflow compared to simpler settings.
- **Higher dependency complexity correlates with worse performance**, suggesting models struggle to correctly reuse and integrate prior functions as structure grows.

## How is this similar to GALILEO?

- Shared emphasis on **multi-turn evaluation** where error accumulation and context carryover matter.
- Uses **structure-aware analysis** (dependency trees) analogous to analyzing how task structure or “pressure” over turns affects robustness.

## How is this different from GALILEO?

- Different domain: **code generation / software engineering workflows** rather than conversational robustness, sycophancy, or drift under manipulation.
- Focuses on **reuse and modular integration**; GALILEO’s core concerns are behavioral robustness (e.g., consistency under pressure, drift, safety failures) rather than functional composition correctness.

## Where GALILEO is stronger / cleaner (if true)

- GALILEO’s protocols (as positioned) can more directly connect to *behavioral* failure modes (drift, compliance, truthfulness) and time-to-failure style analyses, whereas CodeFlowBench is domain-specific to code.

## Where GALILEO is weaker / needs to improve

- CodeFlowBench’s **explicit dependency-structure labeling** provides a concrete, mechanistic axis (dependency complexity) that could inspire more granular structural annotations/controls in GALILEO-style multi-turn setups.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding a **structure/complexity covariate** for multi-turn episodes (a “dependency/constraint graph” analog) and report performance vs that complexity.
- [ ] In related work, cite CodeFlowBench as evidence that **multi-turn, iterative settings reveal degradation** not visible in single-turn evaluation.

## Quotes / details to potentially cite

- Abstract framing: “implementing new functionality by reusing existing functions over multiple turns” as the core capability being benchmarked.
- Key empirical claim (abstract): “significant performance degradation in multi-turn codeflow scenarios” and “performance inversely correlates with dependency complexity.”
