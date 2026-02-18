# O-Researcher: An Open Ended Deep Research Model via Multi-Agent Distillation and Agentic RL

- Year: 2026
- Venue: arXiv
- Authors: Yi Yao, He Zhu, Piaohong Wang, Jincheng Ren, Xinlong Yang, Qianben Chen, Xiaowan Li, Dingfeng Shi, Jiaxian Li, Qiexiang Wang, Sinuo Wang, Xinpeng Liu, Jiaqi Wu, Minghao Liu, Wangchunshu Zhou
- URL: https://arxiv.org/abs/2601.03743
- BibTeX key (if we add it): ya o2026oresearcher
- Tags: agents, deep-research, synthetic-data, tool-use, rlaif, distillation

## One-sentence takeaway

A multi-agent pipeline generates tool-using “deep research” trajectories for SFT, then applies an agentic RL/RLAIF stage to improve open models on deep-research report benchmarks.

## What problem does it solve?

- Closed/open LLM capability gaps are often driven by access to high-quality proprietary training data; the paper targets *creating* scalable, “research-grade” instruction/trajectory data that teaches long-horizon, tool-integrated research behavior.

## What is the core method / protocol?

- Multi-agent *report generation workflow*:
  - A **planner** decomposes an open-ended query into orthogonal sub-queries.
  - Multiple agents run tool-integrated Plan→Execute→Observe loops in parallel to produce **sub-query reports**.
  - A **summarizer/fusion** model aggregates sub-reports into the final report.
  - All traces (tool calls + intermediate text) are concatenated into a supervised “reasoning trace” for training.
- Data construction details (as reported):
  - Seed ~5,000 open-ended queries from a mix of existing datasets + LLM-synthesized topics.
  - After filtering, retain **3,500+** “premium” instruction-response pairs.
- Quality assurance (rejective sampling pipeline):
  - Oversample ~3 candidate trajectories/query.
  - Hard filters include: structural completeness, context length (mentions 64k), minimum complexity (**≥10 reasoning steps** and **≥5 tool-use actions**), language/format consistency.
  - LLM-as-a-judge (mentions Qwen3-based) for semantic filtering.
  - Human spot-checking that can trigger regeneration.
- Structured serialization:
  - Uses an XML-style schema with explicit tags for decomposition, thoughts/plans, tool calls, tool outputs, intermediate subtask answers, and final answer.

## What are the key metrics?

- Not fully captured in the excerpted HTML; the paper claims SOTA on a “major deep research benchmark” and compares across multiple model scales.
- Protocol-level metrics implied by their pipeline include trajectory depth (steps), tool-action counts, and judge/human quality screening outcomes.

## What are the main results?

- Claims that SFT+RL on their synthetic trajectories closes the gap vs closed-source systems and achieves new SOTA among open models on deep-research benchmarks (exact numbers not in the fetched sections).

## How is this similar to GALILEO?

- Both care about **multi-turn/long-horizon** behaviors and **tool-integrated** agentic workflows.
- Their emphasis on *trajectory-level supervision* (not just final answers) overlaps with GALILEO’s likely need to reason about and evaluate behavior over turns.

## How is this different from GALILEO?

- O-Researcher is primarily a **training data + training method** paper (multi-agent distillation + RL) for building deep-research models.
- GALILEO (rapid related-work focus) is typically more about **evaluation protocols/metrics** for instability/robustness or controlled comparisons; this paper’s evaluation framing is benchmark-performance-centric.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO targets robustness/instability measurement, it can be cleaner on *causal controls* and *failure-mode metrics* than benchmark-only improvements.

## Where GALILEO is weaker / needs to improve

- If GALILEO lacks a scalable trajectory-generation pipeline, this paper is a concrete blueprint for producing large quantities of deep, tool-using traces with quality gates.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adopting (or citing) a **rejective sampling** recipe for collecting high-depth agent trajectories: explicit minima on reasoning depth and tool-action count.
- [ ] If GALILEO needs training data, cite the **planner→parallel tool users→summarizer** decomposition as a scalable template.
- [ ] If GALILEO is evaluation-centric, use this as motivation: “training claims SOTA, but robustness over time/turns remains under-measured.”

## Quotes / details to potentially cite

- Motivation: performance gap attributed to “disparities in access to high-quality training data.”
- Pipeline numbers: “seed set of 5,000 queries … yielded 3,500+ premium instruction-response pairs.”
- Filtering constraints: minimum “10 reasoning steps” and “5 distinct tool-use actions” (as hard rejection criteria).
