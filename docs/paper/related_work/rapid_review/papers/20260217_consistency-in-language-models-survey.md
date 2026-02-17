# Consistency in Language Models: Current Landscape, Challenges, and Future Directions

- Year: 2025
- Venue: ICML 2025 Workshop on Reliable and Responsible Foundation Models (arXiv)
- Authors: Jekaterina Novikova; Carol Anderson; Borhane Blili-Hamelin; Domenic Rosati; Subhabrata Majumdar
- URL: https://arxiv.org/abs/2505.00268
- BibTeX key (if we add it): Novikova2025ConsistencySurvey
- Tags: survey, consistency, robustness, evaluation

## One-sentence takeaway

A compact survey/taxonomy of *behavioral consistency* in LMs (logical + informal notions), arguing that the field lacks shared definitions and high-quality benchmarks—useful framing for positioning consistency as a reliability prerequisite.

## What problem does it solve?

- Research on “consistency” in LMs is fragmented: different papers use incompatible definitions, tasks, and metrics, making it hard to compare methods or make deployment-relevant claims.
- Without standard evaluation, we risk overestimating SOTA performance and underestimating harms from inconsistent behavior in high-stakes use.

## What is the core method / protocol?

- Survey (2019–2025) focused on text-only LMs.
- Proposes an organizing taxonomy of **behavioral consistency**:
  - **Logical/formal consistency** (following formal logic properties), including (via prior work) negational, symmetric, transitive, additive consistency.
  - **Nonlogical/informal consistency** covering notions like semantic consistency, moral/norm consistency, informational/factual consistency, explanation self-consistency vs faithfulness.
- Reviews how consistency has been analyzed across tasks (Q&A, summarization, NLI, etc.), and highlights gaps (terminology, metrics, benchmark quality, model/data access).

## What are the key metrics?

- This paper is primarily taxonomy/survey; it does not introduce a single new metric.
- Catalogs metric families used in prior work:
  - Logical consistency checks (negation/symmetry/transitivity/additivity constraints)
  - Semantic equivalence invariance (same meaning → same prediction)
  - Factual/faithfulness consistency in summarization (often human-annotated)
  - “Self-consistency” of explanations (stability across input variations), contrasted with faithfulness.

## What are the main results?

- Consistency is a prerequisite for trustworthy deployment, but **there is no standard approach** to assess it across domains/tasks.
- Calls out major pain points:
  - Terminology mismatch (authors define “consistency” differently or not at all)
  - Metric fragmentation and limited benchmark comparability
  - Need for stronger, higher-quality benchmarks and interdisciplinary approaches that preserve utility.

## How is this similar to GALILEO?

- Shares the core motivation: **reliable behavior across interactions/settings** is essential and current LMs can be inconsistent.
- Provides language to justify why consistency/instability should be evaluated explicitly rather than assumed from aggregate accuracy.

## How is this different from GALILEO?

- This is a survey/taxonomy; it does not provide a concrete, GALILEO-style protocol focused on *multi-turn pressure, drift vs revision controls, or recovery dynamics*.
- Focuses broadly on consistency across many NLP task contexts, not specifically on interactive social-pressure settings.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO includes paired conditions (pressure vs evidence) and multi-turn dynamics, it offers a more operational, decision-relevant evaluation than the broad survey framing.
- A concrete protocol + metrics for *time-to-failure / oscillation / recovery* would be a strong complement to this survey’s call for better benchmarks.

## Where GALILEO is weaker / needs to improve

- If our paper’s terminology is not crisp, this survey is a warning sign: we should define “consistency” precisely and position GALILEO within (or alongside) the logical vs informal taxonomy.

## Action items for GALILEO (experiments / method / writing)

- [ ] In related work, explicitly situate GALILEO as measuring **behavioral consistency in interactive settings**, and clarify which subtype(s) we measure (logical? semantic? norm? pressure-induced belief drift?).
- [ ] Add a short “terminology” paragraph: define consistency vs stability vs faithfulness vs self-consistency, and explain what we do *not* claim to measure.
- [ ] Use the survey’s motivation (“no standard approaches”) to justify our benchmark/protocol contribution.

## Quotes / details to potentially cite

- Abstract-level definition: consistency as “expressing similar meanings in similar contexts and avoiding contradictions.”
- Claim: advanced LMs “struggle with consistency” and there are “no standard approaches to assessing model consistency,” risking overestimation of SOTA and underestimation of harms.
