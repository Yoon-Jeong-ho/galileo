# Step-DeepResearch Technical Report

- Year: 2025
- Venue: arXiv (Technical Report)
- Authors: Chen Hu, Haikuo Du, Heng Wang, Lin Lin, Mingrui Chen, Peng Liu, Ruihang Miao, Tianchi Yue, Wang You, Wei Ji, Wei Yuan, Wenjin Deng, Xiaojian Yuan, Xiaoyun Zhang, … (many authors)
- URL: https://arxiv.org/abs/2512.20491
- BibTeX key (if we add it): stepdeepresearch2025
- Tags: deep-research, agent-training, evaluation, rl, sft, chinese-benchmark

## One-sentence takeaway

Step-DeepResearch trains a single-agent “deep research” model end-to-end using synthetic data decomposed into atomic capabilities plus progressive SFT→RL with a checklist-style judge, and introduces ADR-Bench (Chinese) to evaluate realistic open-ended research.

## What problem does it solve?

- Benchmarks like multi-hop QA / BrowseComp under-represent real “open-ended research”: latent intent recognition, long-horizon planning, multi-turn tool use, cross-source verification, and coherent report writing.
- Practical need for *cost-effective* deep-research agents that don’t rely on heavy workflow orchestration.

## What is the core method / protocol?

- **Atomic-capability decomposition + data synthesis**: build targeted training data for planning, info seeking, reflection/error correction, summarization, verification, and report writing.
- **Progressive training pipeline**:
  - “Agentic mid-training” (capability shaping)
  - Supervised fine-tuning (SFT)
  - Reinforcement learning (RL)
- **Checklist-style judger / reward design** to improve robustness across scenarios.
- System framing emphasizes internalizing an expert-like research loop rather than hardcoding multi-agent workflows.

## What are the key metrics?

- **Scale AI ResearchRubrics** score (ternary rubric compliance), reported as **61.42** for the 32B model.
- **ADR-Bench**: expert human evaluation with **Elo-style ratings** across multiple quality dimensions (Chinese-domain, realistic scenarios).
- Cost-efficiency comparisons (reported qualitatively/with a cost-performance frontier figure).

## What are the main results?

- Step-DeepResearch (32B) reaches **61.42** on ResearchRubrics, described as comparable to proprietary “DeepResearch” services while being cheaper to deploy/run.
- On ADR-Bench, expert Elo ratings reportedly outperform similarly sized/open models and approach closed-source systems.
- Claims robustness gains from the progressive training + checklist judger.

## How is this similar to GALILEO?

- Both care about **long-horizon agent competence** beyond single-turn QA: planning, verification, and producing structured outputs.
- Uses an explicit **capability decomposition** framing (useful for designing evaluations and ablations).

## How is this different from GALILEO?

- Focus is specifically on **deep research/report generation** agents and their training/evaluation; GALILEO’s core scope may be broader than research-style tasks.
- Emphasizes **end-to-end internalization** over workflow orchestration; if GALILEO uses more explicit components, this provides a contrasting baseline.
- Introduces a **Chinese-domain** benchmark (ADR-Bench) and cost-efficiency angle.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO has clearer, task-grounded definitions and more transparent evaluation protocols, it can be stronger than vendor-style rubric scoring.
- If GALILEO emphasizes reproducible datasets and open evaluation artifacts, it may be cleaner than an Elo-heavy expert evaluation.

## Where GALILEO is weaker / needs to improve

- If GALILEO lacks **application-driven** open-ended research benchmarks (esp. multilingual/Chinese), ADR-Bench-style scenarios could expose gaps.
- If GALILEO doesn’t explicitly train/evaluate **cross-source verification** and *report quality*, this paper provides concrete dimensions to incorporate.

## Action items for GALILEO (experiments / method / writing)

- [ ] Map GALILEO’s existing tasks/metrics onto an **atomic-capabilities** taxonomy (planning, seeking, verification, reflection, writing) and identify missing capabilities.
- [ ] Add a **checklist-style judge** variant (or rubric decomposition) to test robustness vs. a single scalar judge.
- [ ] Consider an **Elo-style** pairwise eval protocol for subjective report quality (with guardrails for reproducibility).
- [ ] Identify whether a **Chinese/multilingual** “deep research” slice is needed for GALILEO’s evaluation story.

## Quotes / details to potentially cite

- “Search is not research.” (they explicitly argue multi-hop QA incentives bias agents toward retrieval-heavy behavior rather than coherent synthesis)
- Deep Research reframed as “long-horizon decision-making over a set of atomic capabilities” (planning, information gathering/verification, reflection, writing).
- Reported numbers: 32B model; ResearchRubrics **61.42**; introduces **ADR-Bench** with expert Elo ratings.
