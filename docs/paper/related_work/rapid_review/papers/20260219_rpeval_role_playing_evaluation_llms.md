# Role-Playing Evaluation for Large Language Models (RPEval)

- Year: 2025
- Venue: arXiv
- Authors: Yassine El Boudouri, Walter Nuninger, Julian Alvarez, Yvan Peter
- URL: https://arxiv.org/abs/2505.13157
- BibTeX key (if we add it): rpeval2025
- Tags: role-play, evaluation, benchmark, in-character-consistency, moral-alignment

## One-sentence takeaway

RPEval is a single-turn, fully-automatable benchmark for evaluating LLM role-playing along emotional understanding, decision-making, moral alignment, and in-character consistency.

## What problem does it solve?

- Role-play “quality” is hard to evaluate reproducibly: human evaluation is expensive/variable, and model-judge evaluation can be biased by the judge.
- Existing evaluations often require multi-turn interaction, which is slower/costly and complicates automation.

## What is the core method / protocol?

- Construct a benchmark of **single-turn** role-play tasks: a system/dialogue prompt defines the character, then a single message arrives from another character.
- Evaluate responses on 4 dimensions (chosen to be cheaply/verifiably scored in a single turn):
  - **Emotional understanding:** select one of 13 predefined emotion labels.
  - **Decision-making:** binary **yes/no** choice consistent with persona/goals.
  - **Moral alignment:** binary **yes/no** consistent with persona’s ethical frame.
  - **In-character consistency (knowledge boundaries):** avoid “out-of-context knowledge leakage” (e.g., shouldn’t know modern facts in medieval setting).
- Data construction pipeline (high-level):
  - Generate character profiles and then character descriptions (3125 descriptions generated using GPT-4o).
  - Generate scenarios per character per dimension (total reported scenarios generated: 18,850).
  - Crowd-annotate expected answers for emotion + decision/moral via majority vote; filter by agreement thresholds; final benchmark retained 9,018 scenarios over 3,061 characters.

## What are the key metrics?

- Per-item **binary correctness** (1/0) per dimension.
- Report average accuracy (mean of binary scores) across scenarios.
- Robustness/stability check: repeat runs (n=6) and report std-dev of scores (reported ~0.89% for average).

## What are the main results?

- Baselines on GPT-4o, Gemini-1.5-Pro, and Llama 3.2 1B.
- Gemini-1.5-Pro highest average (reported 62.24%), strong on decision/moral and consistency.
- GPT-4o reported to do well on decision/moral but extremely poorly on in-character consistency (reported 5.81%), often answering OOD questions rather than refusing/deflecting in-character.

## How is this similar to GALILEO?

- “Behavioral evaluation under constraints”: checks whether a model follows a specified role/persona and respects boundaries.
- Highlights that aggregate scores can hide failure modes (e.g., strong reasoning but poor boundary adherence), which is a useful lens for evaluation design.

## How is this different from GALILEO?

- RPEval is **single-turn** and tightly structured (emotion label / yes-no / leakage checks), whereas many agentic/system evaluations focus on multi-step interactions and richer outcomes.
- Focus is specifically on **role-play persona fidelity**, not general task success or tool-using agent behavior.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO emphasizes end-to-end task validity or multi-turn dynamics, it likely captures failure modes that single-turn benchmarks cannot (e.g., drift over time, recovery after a misstep).

## Where GALILEO is weaker / needs to improve

- If GALILEO does not already have explicit “persona boundary”/knowledge-leakage checks, RPEval suggests these can be operationalized cheaply (keyword/boundary-style verification) and added as a dimension.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding a **knowledge-boundary / out-of-context leakage** sub-metric for any “role/policy constrained” evaluations.
- [ ] When presenting results, separate dimensions explicitly (avoid single aggregate score hiding “consistency” failures).
- [ ] Consider a small “single-turn smoke test” suite to complement longer, more expensive multi-turn evaluations.

## Quotes / details to potentially cite

- RPEval evaluates role-play across “four key dimensions: emotional understanding, decision-making, moral alignment, and in-character consistency.”
- Design choice: single-turn interactions for “cost efficiency, speed, and reproducibility.”
- Dataset scale after filtering: 9,018 scenarios across 3,061 characters.
