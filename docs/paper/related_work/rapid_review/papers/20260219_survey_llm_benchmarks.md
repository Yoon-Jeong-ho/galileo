# A Survey on Large Language Model Benchmarks

- Year: 2025
- Venue: arXiv
- Authors: Shiwen Ni, Guhong Chen, Shuaimin Li, Xuanang Chen, Siyi Li, Bingli Wang, Qiyao Wang, Xingjian Wang, Yifan Zhang, Liyang Fan, Chengming Li, Ruifeng Xu, Le Sun, Min Yang
- URL: https://arxiv.org/abs/2508.15361
- BibTeX key (if we add it): Ni2025SurveyLLMBenchmarks
- Tags: survey, benchmarks, evaluation, data-contamination, bias, reliability

## One-sentence takeaway

A broad survey that categorizes 283 LLM benchmarks and highlights recurring evaluation failure modes (contamination, cultural/language bias, weak assessment of process credibility and dynamic environments).

## What problem does it solve?

- Provides a structured map of the fast-growing LLM benchmark landscape (what exists, how to categorize it).
- Summarizes common issues in benchmark-based evaluation that can inflate/warp conclusions.

## What is the core method / protocol?

- Survey + taxonomy: 283 representative benchmarks grouped into:
  - General capabilities (e.g., core linguistics, knowledge, reasoning)
  - Domain-specific (e.g., natural sciences, HSS, engineering/technology)
  - Target-specific (e.g., risks, reliability, agents)
- Discussion of benchmark pitfalls + suggested design paradigms for future benchmarks.

## What are the key metrics?

- Not a single metric paper; discusses evaluation issues qualitatively.
- Key “evaluation quality” axes surfaced:
  - Data contamination / train–test leakage
  - Cultural & linguistic bias / unfairness
  - Lack of process-credibility evaluation (how an answer was produced)
  - Lack of dynamic / interactive environment evaluation

## What are the main results?

- A consolidated categorization of benchmark types and representative areas.
- A clear set of critique points about why benchmark leaderboards can mislead (inflated scores, bias, missing robustness/process/dynamics).

## How is this similar to GALILEO?

- Same meta-goal: evaluation that actually reflects “capability” rather than artifacts.
- Directly supports GALILEO’s motivation: static one-shot benchmarks miss important failure modes; robustness in dynamic / multi-turn settings matters.

## How is this different from GALILEO?

- This is a survey/taxonomy paper; it does not propose a concrete new evaluation protocol with auditable runs.
- Focuses on breadth across benchmarks; GALILEO is (presumably) a targeted evaluation methodology / analysis framework.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO includes strict auditing and leakage controls, it can be positioned as directly addressing the “inflated scores / contamination” critique.
- If GALILEO evaluates multi-round dynamics and failure trajectories, it addresses the “dynamic environments” gap.
- If GALILEO includes per-example traces / flip or failure-turn analysis, it partially operationalizes “process credibility” via observable dynamics.

## Where GALILEO is weaker / needs to improve

- Need to explicitly state how GALILEO mitigates data contamination risks (dataset provenance, dedup checks, prompt leakage considerations).
- Need to proactively address cultural/language bias (scope statement; avoid overclaiming generality).

## Action items for GALILEO (experiments / method / writing)

- [ ] Related work: add a short paragraph citing this survey for (i) benchmark proliferation and (ii) common pitfalls (contamination, bias, missing dynamic/process evaluation) as motivation for GALILEO.
- [ ] Paper framing: explicitly claim which “missing evaluation dimensions” GALILEO covers (dynamic environments, reliability/agents) and which it does not (e.g., multilingual fairness unless tested).
- [ ] Methods appendix/footnote: add a crisp statement about leakage/contamination mitigation for GALILEO’s tasks/data.

## Quotes / details to potentially cite

- They categorize “283 representative benchmarks” into three groups: “general capabilities, domain-specific, and target-specific.”
- Noted benchmark problems: “inflated scores caused by data contamination”, “unfair evaluation due to cultural and linguistic biases”, and “lack of evaluation on process credibility and dynamic environments.”
