# PingPong: A Benchmark for Role-Playing Language Models with User Emulation and Multi-Model Evaluation

- Year: 2024
- Venue: arXiv
- Authors: Ilya Gusev
- URL: https://arxiv.org/abs/2409.06820
- BibTeX key (if we add it): gusev2024pingpong
- Tags: role-playing, benchmark, multi-turn, user-emulation, llm-as-a-judge, ensemble

## One-sentence takeaway

A dynamic, multi-turn role-play benchmark that uses an LLM “interrogator” to emulate users and an ensemble of LLM judges to score character consistency, entertainment, and fluency, with validation against human ratings.

## What problem does it solve?

- Role-play/chat “character” models are hard to evaluate with static, single-turn benchmarks.
- Human evaluation is expensive; static public tests risk contamination/memorization.
- Need a repeatable way to compare many models on interactive role-play quality.

## What is the core method / protocol?

- Three roles (separate LMs):
  - **Player**: model being evaluated; instructed with a **character card** (persona).
  - **Interrogator**: a user-emulator LM that drives a **multi-turn** conversation in a given **situation** (goal/context), with questions generated dynamically (not pre-authored).
  - **Judge(s)**: one or more LMs that score the resulting dialogue.
- Scoring is **single-point** (not pairwise vs a baseline) and uses multiple criteria.
- Uses **multi-model evaluation** by averaging scores from multiple judge models (reported as improving correlation vs single judge).

## What are the key metrics?

- **Character consistency**: responses align with assigned persona.
- **Entertainment value**: engaging/fun/interesting role-play.
- **Language fluency**: grammaticality / naturalness / quality.
- Also tracks whether the player refused to answer.

## What are the main results?

- Evaluated **40+ models** in **English and Russian**.
- Each model: **64 conversations** covering **8 characters × 8 situations**.
- Automated judge scores show **strong correlation** with human annotations (reported via Spearman correlations).
- Ensembling multiple judges correlates better with humans than a single-judge setup.
- Reported finding: fine-tuning for **creative writing** can improve role-play ability.

## How is this similar to GALILEO?

- Treats evaluation as an **interactive, multi-turn** process rather than static prompts.
- Uses **simulated users** (interrogator) to generate more realistic conversational trajectories.
- Uses LLM-based evaluation signals (judge) rather than only task-accuracy metrics.

## How is this different from GALILEO?

- Target domain is **entertainment role-play** (persona adherence + entertainment), not general task assistance.
- Emphasizes **dynamic user question generation** and **judge ensembling** as core benchmark design choices.
- Evaluates (at least in the described setups) on fixed grids of **characters × situations**, not open-ended real-user logs.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO is oriented around real tasks/outcomes, it may offer clearer downstream utility metrics beyond “entertainment value”.
- If GALILEO has stronger controls for safety/grounding, role-play entertainment benchmarks may under-emphasize these aspects.

## Where GALILEO is weaker / needs to improve

- If GALILEO relies on single-turn or static test sets, PingPong highlights the value of **multi-turn + dynamic** evaluation to reduce memorization/contamination.
- If GALILEO uses a single judge, PingPong suggests **judge ensembles** can improve robustness / human alignment.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding a **dynamic user-emulator** condition to at least one GALILEO evaluation suite (multi-turn trajectories, not fixed prompts).
- [ ] Test **multi-judge aggregation** (simple averaging as a baseline) and measure correlation/stability vs single-judge.
- [ ] If relevant, explicitly motivate contamination resistance: dynamic generation as a mitigation.

## Quotes / details to potentially cite

- “three main components: a player model … an interrogator model … and a judge model ensemble” with metrics “character consistency, entertainment value, and language fluency.”
- Scale choice rationale: switched to **5-point Likert** (vs 10-point) to match human annotations and reduce cognitive load.
- Design traits called out as novel: **multi-turn**, **dynamic** (generated interrogator questions), **multi-model** judges.
