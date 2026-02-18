# Foundation Models for Geospatial Reasoning: Assessing Capabilities of Large Language Models in Understanding Geometries and Topological Spatial Relations

- Year: 2025
- Venue: IJGIS (International Journal of Geographical Information Science); arXiv:2505.17136
- Authors: Song Gao et al.
- URL: https://arxiv.org/abs/2505.17136
- BibTeX key (if we add it): gao2025geospatial
- Tags: geospatial, spatial-reasoning, topology, wkt, llm-eval

## One-sentence takeaway

LLMs can do moderately well (>0.6 accuracy) at inferring basic topological relations between vector geometries when geometries are serialized as WKT and prompts are carefully designed, but performance is brittle and benefits from task- and type-specific context.

## What problem does it solve?

- Evaluates whether general-purpose LLMs preserve/understand *geometric structure* and *topological spatial relations* (e.g., contains/intersects/disjoint) when geospatial vector data are provided in text form.
- Clarifies which interaction style works better for geospatial QA: embedding-based retrieval vs prompt-engineered reasoning vs “everyday language” descriptions.

## What is the core method / protocol?

- Represent geometries using **WKT (Well-Known Text)** and query LLMs about relations between two geometries (topological predicates).
- Compare three approaches:
  - **Geometry embedding-based** approach (embed geometry representations; then use retrieval/QA pipeline).
  - **Prompt engineering-based** approach (few-shot / structured prompts to induce correct topological inference).
  - **Everyday language-based** evaluation (vernacular spatial descriptions → map to formal relations).
- Evaluate multiple models (reported: GPT-3.5-turbo, GPT-4, DeepSeek-R1-14B).

## What are the key metrics?

- Accuracy on identifying / inferring **topological spatial relations** between two geometries.
- (Qualitative) ability to:
  - understand **inverse relations** (e.g., if A contains B, then B is within A)
  - use LLM-generated geometry descriptions to help retrieval
  - translate vernacular place descriptions into formal topological relations

## What are the main results?

- Embedding-based and prompt-engineering approaches with GPT models reach **>0.6 average accuracy** on topological relation identification.
- Best reported: **GPT-4 + few-shot prompting** at **>0.66 accuracy** for topological relation inference.
- Adding geometry-type / place-type context in prompts *can* improve inference, but is instance-dependent.
- GPT-4 shows some ability to map vernacular descriptions to formal topological relations.

## How is this similar to GALILEO?

- Adjacent at the “representation → reasoning fidelity” level: tests whether a model can reliably reason over a structured serialization (WKT) without losing critical relational constraints.
- Reinforces the idea that **prompting/protocol** choices can dominate apparent capability, which is relevant for designing robust evaluations.

## How is this different from GALILEO?

- Not about social pressure, multi-turn drift, belief revision, or recovery dynamics.
- Primarily a **single-task accuracy** study in a specific domain (geospatial topology), not a multi-turn robustness protocol.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO targets multi-turn stability under pressure, its contribution is orthogonal and likely more central to LLM reliability than domain-specific topology QA.

## Where GALILEO is weaker / needs to improve

- If GALILEO claims broad “structured reasoning” competence, it may need explicit demonstrations on non-text symbolic serializations (like WKT) to avoid overgeneralizing from purely linguistic tasks.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider a small **structured-serialization stress test** appendix: feed a compact symbolic artifact (e.g., WKT-like strings or relation tables) and measure relation inference accuracy + sensitivity to prompt framing.
- [ ] If we discuss robustness of evaluators/prompting, cite this as a domain example where **few-shot prompting materially changes measured reasoning accuracy**.

## Quotes / details to potentially cite

- “Embedding-based and prompt engineering-based approaches … can achieve an accuracy of over 0.6 on average …” (abstract)
- “GPT-4 with few-shot prompting achieved … over 0.66 accuracy on topological spatial relation inference.” (abstract)
