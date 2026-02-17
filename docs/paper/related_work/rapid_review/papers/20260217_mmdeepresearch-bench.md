# MMDeepResearch-Bench: A Benchmark for Multimodal Deep Research Agents

- Year: 2026
- Venue: arXiv
- Authors: Peizhou Huang; Zixuan Zhong; Zhongwei Wan; Donghao Zhou; Samiul Alam; Xin Wang; Zexin Li; Zhihao Dou; Li Zhu; Jing Xiong; Chaofan Tao; Yan Xu; Dimitrios Dimitriadis; Tuo Zhang; Mi Zhang
- URL: https://arxiv.org/abs/2601.12346
- BibTeX key (if we add it): mmdeepresearchBench2026huang
- Tags: agents, multimodal, web-search, long-horizon, evaluation

## One-sentence takeaway

A 140-task benchmark for *end-to-end* multimodal “deep research agents” that produce citation-rich reports from image+text task bundles, with a three-part evaluation pipeline to score writing quality, citation/evidence alignment, and text–image integrity.

## What problem does it solve?

- Existing “deep research agent” benchmarks are mostly text-only, while multimodal benchmarks are often short-form QA; this misses whether agents can use *visual evidence* (charts/tables/diagrams/screenshots) correctly while doing multi-step retrieval and writing a long-form report.
- Evaluating open-ended, long-horizon, tool-using agents is hard because there may not be a single gold answer; a cited report is the main observable artifact.

## What is the core method / protocol?

- **MMDR-Bench dataset**: 140 expert-crafted tasks spanning **21 domains**.
  - Each task is an **image–text bundle** (query + a small set of images that must be interpreted and integrated).
  - Two regimes:
    - **Daily**: lighter, “everyday” usage; visuals like screenshots / UI captures.
    - **Research**: heavier analysis; visuals like charts, tables, diagrams.
- **Evaluation pipeline (report-level)** with three modules:
  - **FLAE** (Formula–LLM Adaptive Evaluation): report quality (readability, insightfulness, structural completeness) via a mix of auditable formula features + LLM-judge scoring with task-adaptive weighting.
  - **TRACE** (Trustworthy Retrieval-Aligned Citation Evaluation): parses citations, extracts claims, aligns claims to cited URLs, and judges support/coverage (including a visual-evidence component).
  - **MOSAIC** (Multimodal Support-Aligned Integrity Check): checks text–visual consistency / integrity, gated to run only if earlier text/citation scores clear thresholds.
- Introduces **Visual Evidence Fidelity (VEF)** as a strict pass/fail style constraint to enforce alignment between claims and (textualized) visual ground truth.
- Runs experiments across **25** models/systems and analyzes tradeoffs between prose quality, citation discipline, and multimodal grounding.

## What are the key metrics?

- FLAE overall score (0–100), from fused sub-dimensions:
  - Readability, Insightfulness, Structural Completeness.
- TRACE citation-grounding metrics (paper frames them as fine-grained signals; includes checks like consistency/coverage/fidelity and a **visual evidence fidelity** term).
- MOSAIC integrity score for text–image consistency (gated/conditional evaluation).

## What are the main results?

- Across 25 evaluated systems, results show **systematic tradeoffs**:
  - Strong report writing / prose does **not** imply faithful evidence use.
  - **Multimodal integrity** (using visual artifacts correctly) remains a key bottleneck.

## How is this similar to GALILEO?

- Same meta-problem: evaluating (and diagnosing) **long-horizon, multi-step, tool-using agents** where the final artifact is more than a single answer.
- Emphasizes **process-faithfulness signals** (grounding, integrity) rather than only end-task success.

## How is this different from GALILEO?

- Targets **multimodal deep research reports** with citations; GALILEO (as used in this related-work track) is more centered on multi-turn robustness / drift / pressure / belief stability protocols.
- The evaluation suite is oriented around **report quality + citation auditing + text–image consistency**, not primarily turn-by-turn behavioral stability under adversarial/social pressure.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO focuses on controlled multi-turn perturbation protocols (pressure/drift controls, time-to-failure), it can offer **cleaner causal attribution** than open-ended “deep research” tasks where many confounders exist (retrieval variance, writing style, citation formatting).

## Where GALILEO is weaker / needs to improve

- If GALILEO lacks multimodal tasks, MMDR-Bench highlights a missing dimension: **visual evidence use** and text–image integrity in long-horizon workflows.
- If GALILEO does not audit citations/evidence alignment in generated long-form outputs, TRACE-style claim–source checking is a useful neighbor.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding a *multimodal* variant of a GALILEO protocol: include a chart/table as “ground truth evidence” and test whether models’ multi-turn reasoning preserves correct extraction/interpretation under pressure.
- [ ] Borrow the idea of **gated evaluation stages**: only run expensive integrity checks when basic quality/grounding passes.
- [ ] Add a lightweight “evidence fidelity” check analogous to **VEF** for tasks where visual/structured artifacts matter (even if implemented without images, e.g., structured tables).

## Quotes / details to potentially cite

- “Deep Research Agents (DRAs) generate citation-rich reports via multi-step search and synthesis … missing end-to-end multimodal evidence use.”
- “A benchmark of 140 expert-crafted tasks across 21 domains … each task provides an image–text bundle … citation-grounded report generation.”
- “Experiments across 25 state-of-the-art models reveal systematic trade-offs between generation quality, citation discipline, and multimodal grounding.”
