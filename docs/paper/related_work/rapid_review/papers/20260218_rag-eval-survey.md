# Retrieval Augmented Generation Evaluation in the Era of Large Language Models: A Comprehensive Survey

- Year: 2025
- Venue: arXiv (survey)
- Authors: Aoran Gan (per arXiv listing; full author list not captured in abstract page scrape)
- URL: https://arxiv.org/abs/2504.14891
- BibTeX key (if we add it): gan2025ragevalsurvey
- Tags: rag, evaluation, survey, llm

## One-sentence takeaway

A broad survey that systematizes how to evaluate RAG systems (retrieval + generation) across performance, factuality, safety, and efficiency, and catalogs datasets/frameworks while summarizing evaluation practices in recent RAG work.

## What problem does it solve?

- RAG evaluation is harder than plain LM evaluation because outcomes depend on *both* retrieval and generation, plus dynamic knowledge sources and hybrid failure modes.
- The field uses a fragmented set of metrics/datasets/protocols, making comparisons and progress claims inconsistent.

## What is the core method / protocol?

- Survey + taxonomy of RAG evaluation approaches.
- Categorizes (per abstract):
  - system performance evaluation,
  - factual accuracy evaluation,
  - safety evaluation,
  - computational efficiency evaluation.
- Compiles RAG-specific datasets and evaluation frameworks.
- Includes a meta-analysis of evaluation practices in “high-impact” RAG papers (details not in abstract).

## What are the key metrics?

- Not enumerated in the arXiv abstract page.
- Metric families emphasized (abstract-level):
  - task/system performance,
  - factuality / verifiability,
  - safety,
  - compute/efficiency.

## What are the main results?

- Not a new model/benchmark result paper; contribution is consolidation:
  - a comprehensive mapping of RAG evaluation methods,
  - a categorized dataset/framework inventory,
  - a meta-level picture of what evaluation practices are common in top RAG research.

## How is this similar to GALILEO?

- Both are “evaluation-first” contributions: the central object is an evaluation protocol/metric suite rather than a new base model.
- The survey’s emphasis on *hybrid / multi-component systems* (retrieval+generation) parallels GALILEO’s emphasis on *process/interaction-dependent* failures (multi-round pressure dynamics).
- Useful as a citation when motivating why evaluation needs to cover more than single-turn accuracy (even though this survey is RAG-focused rather than persona-pressure focused).

## How is this different from GALILEO?

- Scope: RAG systems and their components vs. GALILEO’s multi-round robustness-under-pressure protocol.
- Outputs: taxonomy/dataset catalog/meta-analysis vs. a concrete experimental protocol with survival/TOF/recovery metrics.
- Failure modes: retrieval errors / grounding / hallucinations / safety vs. belief/answer stability under repeated questioning + persona pressure.

## Where GALILEO is stronger / cleaner (if true)

- Provides concrete, operational, multi-round metrics (e.g., Survival@r, TOF/Fail@1, recovery) and a matched control baseline for drift, enabling direct measurement of interaction-induced degradation.
- The “conditioning on initially-correct” design is a clean robustness lens that avoids conflating capability with stability.

## Where GALILEO is weaker / needs to improve

- Compared to RAG evaluation practice, GALILEO may need clearer discussion of:
  - factuality/verification-oriented scoring (beyond correctness labels),
  - safety evaluation framing (if claiming safety relevance),
  - compute/efficiency reporting norms.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a brief related-work paragraph positioning GALILEO as complementary to broader LLM-system evaluation surveys: “RAG evaluation has highlighted hybrid, protocol-dependent failures; similarly, multi-turn interaction protocols reveal robustness dynamics not captured by one-shot metrics.” Cite this survey as a representative consolidation.
- [ ] Consider a small “evaluation dimensions” sidebar/table (performance vs robustness vs safety vs efficiency) and place GALILEO explicitly in the “robustness under interaction protocol” cell.

## Quotes / details to potentially cite

- From abstract (paraphrased for now; pull exact wording if we cite): evaluating RAG is uniquely challenging due to hybrid retrieval+generation architecture and dependence on dynamic knowledge sources; survey reviews traditional + emerging evaluation approaches for performance, factual accuracy, safety, and computational efficiency in the LLM era.
