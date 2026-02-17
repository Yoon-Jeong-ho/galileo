# HealthBench: Evaluating Large Language Models Towards Improved Human Health

- Year: 2025
- Venue: arXiv
- Authors: Rahul K. Arora, Jason Wei, Rebecca Soskin Hicks, Preston Bowman, Joaquin Quiñonero-Candela, Foivos Tsimpourlas, Michael Sharman, Meghan Shah, Andrea Vallone, Alex Beutel, Johannes Heidecke, Karan Singhal
- URL: https://arxiv.org/abs/2505.08775
- BibTeX key (if we add it): HealthBenchArora2025
- Tags: medical, safety, multi-turn, benchmark, rubric-eval

## One-sentence takeaway

HealthBench is a 5k-example **multi-turn, open-ended** healthcare benchmark scored by a **rubric-per-conversation** framework authored by hundreds of physicians and graded with a validated model-based evaluator.

## What problem does it solve?

- Prior healthcare LLM evals are often (i) multiple-choice / short-answer and thus misaligned with real clinical dialogue, (ii) weakly validated against expert judgment, and/or (iii) saturated.
- Need a benchmark that jointly measures **quality + safety behaviors** in realistic, multi-turn health interactions.

## What is the core method / protocol?

- Dataset: **5,000** realistic health conversations (user or healthcare professional ↔ model), multi-turn; model must answer the *final user message*.
- Evaluation: **conversation-specific rubrics** written by physicians.
  - Across the dataset: **48,562** unique rubric criteria.
  - Criteria are designed to be self-contained/objective and include both positive and negative items.
  - Each criterion has a nonzero point value in **[-10, +10]**.
- Scoring: a model-based grader checks each criterion independently (full credit if met, else 0), sums points, normalizes by max possible for that conversation; aggregate score is mean across conversations (clipped to [0,1]).
- Stratification: reports scores by high-level **themes** (health task types) and **axes** (behavioral dimensions like accuracy, communication, context awareness).
- Variants:
  - **HealthBench Consensus**: focuses on **34** physician-consensus “most important” behavior dimensions.
  - **HealthBench Hard**: subset intended to remain challenging (reported top scores ~32% at time of release).

## What are the key metrics?

- Primary: overall HealthBench score (mean normalized rubric score; clipped to [0,1]).
- Slice-level: theme/axis breakdowns.
- Reliability angle (as described): agreement of model grader vs physician judgment (reported comparable to physician–physician agreement).

## What are the main results?

- Reported trend: steady progress over ~2 years (e.g., GPT-3.5 Turbo ~16% → GPT-4o ~32%), with more recent rapid improvement (reported o3 ~60%).
- “Small model” point: GPT-4.1 nano reportedly outperforms GPT-4o at much lower cost.
- Human baseline: physicians unassisted are outperformed by recent models; physicians can improve responses when assisted by the same models.
- Remaining gaps highlighted: worst-case performance and context-seeking behavior.

## How is this similar to GALILEO?

- Strong precedent for **multi-turn evaluation** using richer, open-ended outputs rather than MCQ.
- Rubric-based evaluation echoes the broader movement toward **structured, criteria-based grading** (useful if GALILEO uses multi-criteria success/failure definitions).
- Emphasis on safety-relevant behaviors (not just task success).

## How is this different from GALILEO?

- Domain is healthcare conversations; not centered on social-pressure belief drift / evidence-vs-pressure controls.
- The “multi-turn” unit is the conversation context provided to the model for the final turn, rather than an interactive, adversarial multi-round protocol where pressure is applied turn-by-turn.
- Focus is on response quality/safety per scenario, not on **time-to-failure / recovery dynamics** across a fixed-turn interaction.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO explicitly tests **pressure-only drift vs evidence-driven revision**, it offers cleaner causal attribution than a general-purpose rubric benchmark.
- If GALILEO reports trajectory metrics (flip timing, recovery, hazards), it provides dynamics that HealthBench does not target as the primary object.

## Where GALILEO is weaker / needs to improve

- HealthBench shows a credible recipe for **expert-authored, conversation-specific rubrics** at scale (262 physicians), plus validation of a model grader against expert ratings.
- If GALILEO relies on narrower automatic metrics or fewer expert-validated criteria, it may look less “grounded” to applied audiences.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding a small “rubric-eval” appendix: for a subset of scenarios, define a **per-dialogue rubric** (positive + negative criteria) and use it to cross-check our primary metrics.
- [ ] If we use an automated judge, explicitly report **judge–expert agreement** (or at least judge calibration checks) to mirror the trustworthiness story.
- [ ] Borrow the “themes × axes” reporting idea: separate *task/theme* slices from *behavioral dimension* slices for more diagnostic plots.

## Quotes / details to potentially cite

- “HealthBench consists of 5,000 multi-turn conversations…” (Abstract)
- “Responses are evaluated using conversation-specific rubrics created by 262 physicians.” (Abstract)
- “48,562 unique rubric criteria…” (Abstract)
- Hard variant: “HealthBench Hard, where the current top score is 32%.” (Abstract)
