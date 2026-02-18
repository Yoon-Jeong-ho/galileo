# Natural Context Drift Undermines the Natural Language Understanding of Large Language Models

- Year: 2025
- Venue: EMNLP 2025 Findings
- Authors: Yulong Wu; Viktor Schlegel; Riza Batista-Navarro
- URL: https://arxiv.org/abs/2509.01093
- BibTeX key (if we add it): wu2025natural-context-drift
- Tags: context-drift, robustness, evaluation, qa, wikipedia, contamination

## One-sentence takeaway

LLM QA accuracy degrades sharply as Wikipedia passages naturally evolve away from the versions likely seen in pretraining, even when the question and required evidence remain present—suggesting “natural drift” is a real robustness failure mode.

## What problem does it solve?

- Identifies/evaluates a practical robustness issue: real-world reference text (e.g., Wikipedia) is edited over time, so “the same” benchmark passage may drift semantically from what the model memorized/was exposed to during pretraining.
- Proposes an evaluation framework that quantifies performance as a function of *semantic similarity* between inference-time context and training-time content.

## What is the core method / protocol?

- Focus on QA benchmarks whose contexts are sourced from Wikipedia (revision histories available).
- For each original benchmark passage, mine naturally human-edited variants from Wikipedia revision histories (natural perturbations), preserving the answer (they describe an “answers preserving checking” step).
- For a given LLM with known/open training data, retrieve Wikipedia content from its training corpus with the same article title as the edited passage.
- Compute paragraph-level semantic similarity between the edited passage and the closest matching training-corpus Wikipedia content; use max similarity as a proxy for “how close this is to what the model saw”.
- Bin examples into 10 similarity bins; measure QA accuracy per bin; plot trend + confidence intervals.
- Filter out “context-free solvable” items (cases where the model answers correctly using the question alone), to focus on reading-comprehension dependence.
- Compare with human annotators across the same similarity bins.

## What are the key metrics?

- QA accuracy on each dataset (dataset-specific evaluation metric, summarized as correct/incorrect).
- Trend slope of accuracy vs similarity bin (reported as very steep in some cases).
- Wilson score interval (95% CI) per bin.
- Optional: “question-only” accuracy used for filtering.

## What are the main results?

- Across 6 QA datasets and 8 instruction-tuned LLMs (with public training corpora including Wikipedia), accuracy generally declines as semantic similarity to training-corpus Wikipedia decreases.
- Example highlighted in abstract: on BoolQ, average accuracy drops by >30 percentage points from highest- to lowest-similarity bins; some models show extremely steep declines (“slopes exceeding 70”).
- Humans are notably less affected by this semantic drift, maintaining relatively stable accuracy across bins.

## How is this similar to GALILEO?

- Same broad concern: robustness/stability under distribution shift across turns/time, and failure modes that are *not* captured by static single-shot evaluation.
- Methodologically aligned with “stress tests” that vary one factor continuously (here: similarity-to-training) and look for monotonic degradation.

## How is this different from GALILEO?

- This paper is not about multi-turn conversational pressure, persuasion, or sycophancy; it’s about *single-turn* QA robustness under *naturally evolving context documents*.
- The driver is training-data proximity / contamination-like effects (passages close to what was seen during pretraining), rather than social pressure or iterative user attacks.

## Where GALILEO is stronger / cleaner (if true)

- GALILEO (presumably) targets interaction-time robustness and multi-turn dynamics (belief revision vs drift, pressure over rounds), which this paper does not cover.

## Where GALILEO is weaker / needs to improve

- If GALILEO claims “robustness under real-world change”, it likely needs an axis like *natural context evolution* (time-based drift in reference material) as a complementary stressor.
- If GALILEO relies on retrieval / grounding, it should test whether retrieval + reasoning still fails when evidence is phrased differently but semantically equivalent.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a “natural context drift” axis: take a fixed task (QA / verification / policy-following) and perturb *supporting context* using real revision histories (Wikipedia or other doc streams), while keeping the required facts available.
- [ ] Report robustness curves (performance vs similarity) rather than a single score; consider slope/area-under-curve as stability metrics.
- [ ] Add a control for “question-only solvable” cases (filter or separately report) to avoid overestimating comprehension.
- [ ] In writing, explicitly distinguish (a) multi-turn conversational drift, (b) parametric memorization/contamination, and (c) robustness to evolving external knowledge.

## Quotes / details to potentially cite

- Abstract-level result: “LLM performance declines as reading passages naturally diverge from the versions encountered during pretraining—even when the question and all necessary information remains present at inference time.”
- BoolQ example from abstract: “average model accuracy on BoolQ drops by over 30% from the highest to lowest similarity bins”.
