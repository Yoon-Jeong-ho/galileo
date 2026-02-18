# Grade Score: Quantifying LLM Performance in Option Selection

- Year: 2024
- Venue: arXiv
- Authors: Dmitri Iourovitski
- URL: https://arxiv.org/abs/2406.12043v2
- BibTeX key (if we add it): iourovitski2024gradescore
- Tags: evaluation, llm-as-judge, order-bias, consistency, multiple-choice

## One-sentence takeaway

Defines a single scalar “Grade Score” (harmonic mean of order-bias entropy and choice-stability mode frequency) to quantify how reliable an LLM is as a multiple-choice judge under option permutations, and studies prompting/sampling tricks that improve it.

## What problem does it solve?

- When LLMs are used as *multiple-choice selectors/judges* (including for evaluating other models), they can be unreliable due to:
  - **Order/position bias** (e.g., preferring early options)
  - **Inconsistency** across runs or across option permutations
- Prior mitigation (shuffle + re-judge; grade each option separately) can be cumbersome or more expensive.

## What is the core method / protocol?

- Proposes **Grade Score**, combining two components measured over repeated trials under option permutations:
  - **LLM Score (order fairness):** normalized entropy of the *indices* selected across permutations.
  - **Choice Score (stability):** frequency of the *most commonly selected underlying option* (mode) across permutations.
  - **Grade Score:** weighted harmonic mean of LLM Score and Choice Score.
- Experimental setup:
  - Uses OpenAssistant (OASST) dataset rows with a prompt + multiple candidate responses.
  - Generates permutations via a rotation-based “unique permutations” scheme so each option occupies each position once.
  - Evaluates different judge prompts and sampling strategies.
- Introduces **Unrelated Output Sampling**: add an unrelated “random” option from a different prompt as an extra distractor.

## What are the key metrics?

- **Entropy over selected indices** (normalized by log2(n)) → LLM Score in [0,1]
- **Mode frequency over selected options** (mode count / N) → Choice Score in [0,1]
- **Grade Score** = harmonic mean(LLM Score, Choice Score)

## What are the main results?

- **Adding an unrelated option** generally **improves Grade Score** across models and prompts (reported consistently in the paper’s tables).
- Grade Scores vary materially by model and prompt; larger instruction-following models tend to score higher.
- Observes that instruction-following models can partially adapt when explicitly instructed to avoid certain biases (“emergent” bias-targeted compliance).

## How is this similar to GALILEO?

- If GALILEO evaluates candidate actions/answers/options (or uses an LLM to pick among alternatives), this is directly relevant to:
  - diagnosing **position/order effects** in selection-based components
  - designing evaluation protocols robust to **presentation artifacts**

## How is this different from GALILEO?

- Focuses on *multiple-choice judging* under **option permutations**, not end-to-end system behavior.
- Metric is specialized for **selection stability + order bias**, not broader notions of correctness/utility.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO uses pairwise scoring / calibrated reward models / explicit aggregation, it may avoid some position-bias pitfalls inherent in single-shot option selection.

## Where GALILEO is weaker / needs to improve

- Any component that:
  - presents a list of candidates and asks for a single pick
  - does not randomize order / aggregate across permutations
  may be vulnerable to the behaviors quantified here.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a “selection robustness” eval: run candidate selection under multiple order permutations; report entropy (index) + mode frequency (option).
- [ ] Consider using a Grade-Score-like composite metric when comparing selector prompts/strategies.
- [ ] Try “unrelated option” (distractor) ablations to see if it improves discrimination for GALILEO’s selector steps.

## Quotes / details to potentially cite

- Abstract: introduces Grade Score combining “Entropy” (order bias) and “Mode Frequency” (choice stability) for LLMs used as multiple-choice judges.
- Grade Score definition: harmonic mean of LLM Score (normalized entropy over selected indices) and Choice Score (mode frequency over selected options).
- Unrelated Output Sampling: adding an unrelated option can improve Grade Score across prompts/models (as shown in result tables).
