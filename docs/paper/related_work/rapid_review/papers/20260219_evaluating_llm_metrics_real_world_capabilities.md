# Evaluating LLM Metrics Through Real-World Capabilities

- Year: 2025
- Venue: arXiv
- Authors: Justin K. Miller; Wenjia Tang
- URL: https://arxiv.org/html/2505.08253v1
- BibTeX key (if we add it): miller2025evaluating
- Tags: metrics, evaluation, capabilities, human-centered, benchmarks

## One-sentence takeaway

Reframes LLM evaluation around six empirically-derived “real-world utility” capabilities and shows that commonly cited benchmarks under-cover them (especially reviewing work and data structuring), then compares major models on the subset with available benchmarks.

## What problem does it solve?

- Benchmark/leaderboard claims about “best model” often rely on narrow, abstract, or gaming-prone tests (e.g., coding-only, multiple-choice), which may not reflect what people actually use LLMs for in everyday workflows.
- The paper aims to ground evaluation in observed usage: what capabilities matter, and whether existing benchmarks meaningfully test them.

## What is the core method / protocol?

- Derives six “core capabilities” from:
  - Qualitative thematic analysis of high-exposure occupational tasks (from a large worker survey study using an OpenAI “Direct Exposure” style measure).
  - Validation with large-scale real usage data: mapping the top 100 most frequent Claude.ai task categories (from an Anthropic-released dataset) to these capabilities.
- Defines five human-centered evaluation criteria for benchmarks: coherence, accuracy, clarity, relevance, efficiency.
- Maps widely used benchmarks/metrics (as used in model reports) to capabilities and scores their alignment qualitatively against the five criteria.
- For the capabilities where suitable benchmarks exist, compiles reported scores for leading model families and compares.

## What are the key metrics?

- Not a new numeric benchmark; rather:
  - Capability taxonomy: Summarization; Technical Assistance; Reviewing Work; Data Structuring; Generation; Information Retrieval.
  - Benchmark-quality criteria: coherence, accuracy, clarity, relevance, efficiency.
  - Model comparison uses existing public metrics (examples named in the paper):
    - Technical assistance: WebDev Arena (Elo-style)
    - Information retrieval: SimpleQA (accuracy)
    - Summarization: FACTS Grounding / MRCR (groundedness / retrieval-style correctness)
    - Generation: Creative Writing Arena / WritingBench (Elo / critic-validated scores)

## What are the main results?

- Real-world usage is concentrated: a small number of task types dominate prompt volume; the “top 100” tasks account for just over half of observed usage in the referenced dataset.
- Capability prevalence (in their mapping of top tasks) emphasizes:
  - Technical Assistance and Reviewing Work as most common.
  - Data Structuring as relatively rare among top tasks.
- Benchmark coverage gaps:
  - Existing “standard” benchmarks cover Information Retrieval, Technical Assistance, Summarization reasonably.
  - They find essentially no widely adopted benchmarks that directly evaluate Reviewing Work or Data Structuring.
- Model comparison headline (as reported in the paper): Gemini is top or near-top across their selected utility-aligned metrics among surveyed model families.

## How is this similar to GALILEO?

- Shares the premise that evaluation should reflect real user goals and task utility rather than abstract “intelligence” proxies.
- Emphasizes human-centered criteria (clarity, relevance, efficiency) that are often under-modeled in benchmark-centric evaluation.

## How is this different from GALILEO?

- Primarily a meta-evaluation / taxonomy + benchmark-mapping paper, not a new end-to-end evaluation framework with fully specified datasets, tasks, and scoring for the missing capabilities.
- Uses qualitative alignment judgments and existing public scores rather than proposing a unified, reproducible benchmark suite covering all six capabilities.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO provides a concrete, reproducible evaluation protocol (tasks, rubrics, scoring, and reporting), it may be more actionable than qualitative “benchmark coverage” critiques.
- Opportunity for GALILEO: explicitly operationalize “reviewing work” and “data structuring” with measurable tasks and clear grading.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently leans on conventional benchmarks, this paper’s critique suggests risk of misalignment with real-world user value.
- Efficiency (time saved / interaction length / cognitive load) is repeatedly highlighted as missing in benchmarks; GALILEO should ensure it is measured explicitly.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add an evaluation section that motivates capability coverage with usage evidence (e.g., map intended user tasks to a small capability taxonomy).
- [ ] Define (and measure) an explicit efficiency metric (time-to-satisfactory-output, turns-to-solve, or user-rated effort) alongside correctness/quality.
- [ ] Create task suites for the “missing” areas: Reviewing Work (critique + improvement + bug/issue finding) and Data Structuring (schema extraction, table normalization, citation formatting as structured outputs).
- [ ] When citing benchmarks, include a brief “face validity” paragraph using criteria like coherence/relevance to justify benchmark choice.

## Quotes / details to potentially cite

- The six capabilities identified: “Summarization, Technical Assistance, Reviewing Work, Data Structuring, Generation, and Information Retrieval.”
- Human-centered benchmark criteria used throughout: coherence, accuracy, clarity, relevance, efficiency.
- Coverage gap claim: no widely adopted benchmarks found for Reviewing Work and Data Structuring (per their search and mapping).
