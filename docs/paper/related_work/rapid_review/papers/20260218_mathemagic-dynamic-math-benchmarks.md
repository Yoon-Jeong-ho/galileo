# MatheMagic: Generating Dynamic Mathematics Benchmarks Robust to Memorization

- Year: 2025
- Venue: arXiv
- Authors: Dayyán O’Brien; Barry Haddow; Emily Allaway; Pinzhen Chen
- URL: https://arxiv.org/abs/2510.05962
- BibTeX key (if we add it): OBrien2025MatheMagic
- Tags: benchmark, memorization, robustness, math, dynamic-eval, counterfactual-rules, induction-vs-deduction

## One-sentence takeaway

A dynamic, test-time generated “counterfactual arithmetic” benchmark (MatheMagic) that resists memorization/contamination and disentangles deduction (apply stated rules) vs induction (infer rules from examples), showing models often revert to standard math and induction fine-tuning generalizes poorly.

## What problem does it solve?

- Static math benchmarks become unreliable because (i) test sets can leak into training or be explicitly overfit, and (ii) the space of symbols/rules is low-diversity, enabling memorization/pattern matching.
- Need an evaluation protocol where the *surface form* cannot be pre-memorized and where “true reasoning” can be probed under distribution shifts in the underlying rules.

## What is the core method / protocol?

- Define a framework to generate *dynamic, counterfactual* math problems at test time by **changing the rules of arithmetic** (e.g., remapping operators, swapping digits, reversing place values, adding constants, modulo arithmetic, etc.).
- Instances are **procedurally generated** and **seeded** (random seed controls transformation parameters and selection of in-context examples), aiming for both comparability and contamination robustness.
- Two reasoning regimes:
  - **Deduction**: the new rule is explained explicitly; model must apply it.
  - **Induction**: model must infer the rule from worked examples.
- They evaluate across **9 transformations** (Table 1 in the HTML), with multiple seeded test sets (reported as 5×405 examples in this version).
- Additional analyses include error characterization over number of examples, and fine-tuning for induction/generalization.

## What are the key metrics?

- Accuracy on generated test sets, separated by deduction vs induction.
- Breakdown by transformation type (simple operational changes vs multi-step reasoning).
- Error analysis over increasing number of in-context examples (e.g., “reverting to standard math” vs novel/procedural errors).
- Fine-tuning evaluation: in-domain gains vs out-of-domain (generalization across transformations).

## What are the main results?

- Models generally perform **better at deduction than induction**.
- A common failure mode is **reverting to standard arithmetic**, indicating strong priors/overlearned heuristics.
- With more examples, failures shift from “standard-math bias” toward **procedural complexity** errors.
- Fine-tuning on induction tasks yields **modest in-domain improvements** but **weak out-of-domain generalization**, suggesting limited transferable “reasoning skill” under these counterfactual rules.

## How is this similar to GALILEO?

- Shared goal: **robust evaluation** that is less susceptible to contamination/overfitting and better reflects underlying capability.
- Uses **counterfactual / rule-shift** settings to probe whether behavior is driven by memorized patterns vs context-conditioned reasoning.
- The explicit separation of *rule specification* (deduction) vs *rule inference* (induction) is conceptually adjacent to separating “follow the current conversation evidence” vs “default priors/heuristics” in multi-turn settings.

## How is this different from GALILEO?

- Domain is **single-turn math reasoning**, not multi-turn conversational dynamics.
- Focuses on *symbol/rule remappings* with automatically checkable answers, rather than social/multi-turn robustness phenomena (drift, instability, persuasion, recovery).
- Main evaluation axis is induction/deduction under counterfactual arithmetic, not turn-by-turn failure processes.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO targets multi-turn robustness (e.g., time-to-failure, recovery, stability), it addresses a broader class of real deployment risks than counterfactual arithmetic alone.
- Multi-turn protocols can model compounding errors and intervention strategies, which are not central here.

## Where GALILEO is weaker / needs to improve

- If GALILEO uses mostly static test sets, MatheMagic is a good reminder that **dynamic, seeded generation** can improve contamination robustness while preserving comparability.
- The induction-vs-deduction split suggests GALILEO may benefit from explicitly designing conditions that separate “rule told” vs “rule inferred from trajectory.”

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding a *dynamic/seeded* task generation layer (even simple counterfactual remappings) to reduce memorization and enable “fresh” evaluation.
- [ ] Explicitly operationalize **deduction vs induction** analogues in GALILEO protocols (e.g., constraints stated directly vs inferred from prior turns).
- [ ] Add a failure-mode label similar to “revert to prior” (analogous to reverting to standard math) to quantify stubborn-prior vs evidence-driven adaptation.

## Quotes / details to potentially cite

- “MatheMagic … generates math test instances with the interpretations of numbers and operators altered, yet has automatically verifiable answers.” (abstract, arXiv HTML)
- They evaluate 9 transformations (swap_symbols, swap_digits, reverse_place_values, rotate_digits, add_constant, negate_result, modular_arithmetic, trig_substitution, change_base) and report that deduction is easier than induction, with frequent “reversions to standard math.”
