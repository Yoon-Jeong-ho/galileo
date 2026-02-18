# ChartEditBench: Evaluating Grounded Multi-Turn Chart Editing in Multimodal Language Models

- Year: 2026
- Venue: arXiv
- Authors: Manav Nitin Kapadnis; Lawanya Baghel; Atharva Naik; Carolyn Rosé
- URL: https://arxiv.org/abs/2602.15758
- BibTeX key (if we add it): kapadnis2026charteditbench
- Tags: multimodal, chart-editing, multi-turn, evaluation, benchmarks, code-editing

## One-sentence takeaway

ChartEditBench is a 5k-instance benchmark for *multi-turn* chart code editing (not one-shot generation) plus an execution/visual/logic composite evaluator that exposes strong error accumulation and frequent execution failures on data-centric edits.

## What problem does it solve?

- Existing chart benchmarks largely test single-turn chart generation/reconstruction or static chart QA; they miss the common workflow where users iteratively refine plots through a sequence of edits.
- Current evaluation for chart “editing” is brittle: LLM-as-judge scores can be narrow/unstable, while CLIP/embedding similarity can miss semantically important mistakes.

## What is the core method / protocol?

- **Task framing (stateful editing):** given an initial *(code, rendered chart)* pair, apply edits over multiple turns, producing updated code that should match either:
  - **Task 1 (code-to-code via target image):** edit code so the output chart matches a *target modified chart image*.
  - **Task 2 (NL delta):** edit code based on a natural-language instruction describing the desired modification.
- **Dataset:** synthetic generation pipeline producing **5,000** difficulty-controlled multi-turn “modification chains” (default chain length reported as N=5 turns), plus a **430-example human-verified** subset.
  - Uses a broad pool of matplotlib chart types (paper text: 37 types).
  - Modification intents span multiple categories (paper text: 12 modification types; also describes style vs data modification families).
  - Generation includes code execution + rendering validation gates (AST parse; subprocess execution with timeout; must save a figure).
- **Evaluation (composite):**
  - Execution validity (does the code run and render?)
  - Structured “assertions” for instruction following and code requirements
  - Visual similarity component that is more “grounded” than a single scalar embedding similarity (paper describes enumerating concrete visual differences under a rubric).

## What are the key metrics?

- Executability / render success rate.
- Assertion pass rates (instruction-following assertions + general requirement assertions).
- Visual similarity / grounded difference analysis between generated vs target charts.
- (Implicitly) multi-turn degradation / error accumulation across turns.

## What are the main results?

- SOTA MLLMs show **substantial degradation** in multi-turn chart editing due to **error accumulation** and loss of shared context.
- Models do relatively better on **stylistic** edits, but often fail on **data-centric transformations** and exhibit **frequent execution failures**.
- The proposed evaluation aims to be more reliable/diagnostic than pure LLM-judge or CLIP-style similarity.

## How is this similar to GALILEO?

- Evaluates *grounded, intent-aware*, multi-step behavior where the model must maintain state/common ground across turns—similar to agentic workflows GALILEO cares about.
- Emphasizes tool-like correctness (code that executes) and interpretable evaluation signals rather than only subjective judging.

## How is this different from GALILEO?

- Domain is **matplotlib chart editing via code**, with synthetic pipelines; not necessarily the same end tasks/real-world environment GALILEO targets.
- “Grounding” is primarily via *rendered chart images + code* and checks (execution, assertions, pixel/visual similarity), not broader environment interaction.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO’s tasks are based on real artifacts/workflows (vs synthetic generation), it can claim higher ecological validity.
- If GALILEO emphasizes general tool-use beyond plotting code, it may cover a wider slice of agentic competence.

## Where GALILEO is weaker / needs to improve

- If GALILEO lacks long-horizon, multi-turn *edit chains* with explicit state carry-over, ChartEditBench’s framing could be a useful template.
- If GALILEO evaluation relies heavily on LLM-judge scoring, ChartEditBench’s composite “execution + assertions + grounded visual diffs” is a concrete alternative.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding a **multi-turn “edit chain”** split where each turn conditions on the model’s prior output to measure error accumulation.
- [ ] Add **hard validity checks** (execution/compilation/sandbox success) as a first-class metric when tasks involve code or structured artifacts.
- [ ] Use **assertion-style evaluators** (instruction-following + general requirements) to improve interpretability.
- [ ] When outputs are visual/structured, supplement embedding similarity with **explicit difference enumeration** (rubric-based) to localize failures.

## Quotes / details to potentially cite

- “In practice, users iteratively refine visualizations through multi-turn interactions that require maintaining common ground, tracking prior edits, and adapting to evolving preferences.” (abstract)
- ChartEditBench: “5,000 difficulty-controlled modification chains” and a “human-verified subset.” (abstract)
- Evaluation: integrates “execution-based fidelity checks, pixel-level visual similarity, and logical code verification.” (abstract)
