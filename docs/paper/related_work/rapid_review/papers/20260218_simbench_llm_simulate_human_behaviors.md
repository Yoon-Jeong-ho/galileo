# SimBench: Benchmarking the Ability of Large Language Models to Simulate Human Behaviors

- Year: 2025
- Venue: arXiv
- Authors: Tiancheng Hu; Joachim Baumann; Lorenzo Lupo; Nigel Collier; Dirk Hovy; Paul Röttger
- URL: https://arxiv.org/abs/2510.17516
- BibTeX key (if we add it): Hu2025SimBench
- Tags: benchmark, llm-simulation, human-behavior, alignment-tradeoff, demographics

## One-sentence takeaway

SimBench standardizes evaluation of LLMs-as-human-simulators across 20 behavioral datasets and finds current models are weak overall, with strong scaling-by-size effects, no benefit from more inference-time compute, and an instruction-tuning trade-off that hurts high-entropy (pluralistic) human response distributions.

## What problem does it solve?

- Prior work on “LLM simulation of humans” is fragmented: bespoke tasks, inconsistent metrics, and hard-to-compare results.
- Need a reproducible, standardized benchmark to answer when/why simulations succeed/fail, and to compare models and training choices.

## What is the core method / protocol?

- Construct **SimBench** by harmonizing **20 diverse human-behavior datasets** into a **single-turn, self-contained, multiple-choice** evaluation format.
- Evaluate many LLMs on predicting **group-level human response distributions** (not just a single “correct” label).
- Analyze performance vs:
  - model size (scaling)
  - inference-time compute (more decoding / compute-at-test)
  - instruction-tuning / alignment
  - response entropy (consensus vs pluralistic questions)
  - demographic subgroup simulation
  - correlation with other capability benchmarks.

## What are the key metrics?

- Aggregate benchmark score reported on a **0–100** scale (paper reports top score **40.80/100**).
- “Low-entropy” vs “high-entropy” question subsets (proxying consensus vs diversity in human answers).
- Correlations with external benchmarks (notably **MMLU-Pro**, reported **r = 0.939**).

## What are the main results?

- **Overall simulation is limited**: best model only **40.80/100**.
- **Performance scales ~log-linearly with model size**.
- **More inference-time compute does not improve simulation performance** (negative result).
- **Alignment / instruction-tuning trade-off**:
  - improves low-entropy (consensus) questions
  - degrades high-entropy (pluralistic) questions (mode-seeking behavior).
- **Demographic simulation is harder**: models struggle more for specific groups (esp. religious/ideological groups per intro).
- Simulation ability correlates most strongly with **deep, knowledge-intensive reasoning** (MMLU-Pro, **r = 0.939**).

## How is this similar to GALILEO?

- Shares the meta-goal of **stress-testing LLM behavior under realistic evaluation regimes** and avoiding cherry-picked, task-specific metrics.
- The “entropy / plurality” framing is conceptually adjacent to GALILEO’s concerns about **stability vs drift** and **robustness across varying conversational pressure** (though SimBench is single-turn).

## How is this different from GALILEO?

- SimBench is **single-turn multiple-choice**; GALILEO is oriented around **multi-turn robustness**, pressure, and conversational dynamics (sycophancy/persuasion, belief revision vs drift controls).
- SimBench targets **population-level distribution matching** in social/behavioral tasks; GALILEO likely focuses on **interactional failure modes** and longitudinal consistency.
- SimBench’s negative result about inference-time compute is for this particular benchmark setup; may not transfer to **multi-turn** settings.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO evaluates multi-turn dynamics explicitly, it can capture failure modes SimBench cannot (e.g., accumulation of error, susceptibility to persuasion, strategic compliance over rounds).
- If GALILEO uses adversarial/pressure protocols, it may provide clearer evidence about robustness under manipulation than distributional MCQ simulation.

## Where GALILEO is weaker / needs to improve

- If GALILEO lacks a broad, standardized suite spanning many datasets/populations, SimBench is a template for **benchmark modularity + breadth**.
- GALILEO may need an explicit treatment of **response entropy / plurality**: distinguishing “consensus” questions from inherently diverse human preferences.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add an analysis slice analogous to SimBench’s: performance vs **entropy/plurality** of target behaviors (consensus vs diverse preferences) to avoid over-claiming “fidelity” on inherently pluralistic items.
- [ ] If GALILEO uses alignment-tuned vs base models, test for a similar **alignment–pluralism trade-off** (does instruction-following increase mode collapse across rounds?).
- [ ] Consider incorporating (or at least citing) SimBench as the **single-turn** counterpart to GALILEO’s **multi-turn** robustness story.

## Quotes / details to potentially cite

- “SimBench, the first large-scale, standardized benchmark for a robust, reproducible science of LLM simulation.”
- “Even the best LLMs today have limited simulation ability (score: 40.80/100).”
- “Simulation performance is not improved by increased inference-time compute.”
- “Instruction-tuning improves performance on low-entropy (consensus) questions but degrades it on high-entropy (diverse) ones.”
- “Simulation ability correlates most strongly with deep, knowledge-intensive reasoning (MMLU-Pro, r=0.939).”
