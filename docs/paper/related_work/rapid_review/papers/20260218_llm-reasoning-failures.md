# Large Language Model Reasoning Failures

- Year: 2026
- Venue: TMLR 2026 (Survey Certification); arXiv v1
- Authors: Peiyang Song; Pengrui Han; Noah Goodman
- URL: https://arxiv.org/abs/2602.06176
- BibTeX key (if we add it): song2026reasoningfailures
- Tags: survey, reasoning, robustness, prompt-effects, failure-taxonomy

## One-sentence takeaway

A broad survey that systematizes LLM “reasoning failures” with a 2-axis taxonomy (reasoning type × failure type), explicitly carving out *robustness* as inconsistent behavior under small prompt variations.

## What problem does it solve?

- The literature on “LLM reasoning failures” is fragmented across many tasks/domains (logic, math, social reasoning, embodied settings, etc.), making it hard to compare failure modes or identify common root causes.
- The paper aims to unify terminology and provide an organizing framework that connects failure observations to causes and mitigations.

## What is the core method / protocol?

- Survey + taxonomy.
- Axis 1: reasoning types
  - Embodied vs non-embodied; non-embodied subdivides into informal (intuitive) vs formal (logical / symbolic).
- Axis 2: failure types
  - Fundamental failures (intrinsic/architectural; broad across tasks)
  - Application-specific limitations (domain-tied shortcomings)
  - Robustness issues (performance instability under minor variations)
- For each class of failure, they provide definitions, representative examples (appendix), prior work pointers, hypothesized causes, and mitigation directions.
- They also maintain an “awesome list” repo of papers on reasoning failures.

## What are the key metrics?

- Not a new benchmark; metrics vary by the cited sub-literatures.
- The survey explicitly highlights *robustness* framing: sensitivity / inconsistency across “minor variations” of prompts, formats, perspectives, distractors, etc.

## What are the main results?

- Main contribution is conceptual: a structured map of failure modes and where they occur.
- Emphasizes that some failures look like “capability gaps” while others are better understood as *robustness/instability* (apparently good performance that is brittle).

## How is this similar to GALILEO?

- Overlaps with GALILEO’s motivation: models can look strong on headline tasks but fail under small conversational/prompt perturbations.
- Provides a vocabulary for positioning GALILEO as studying a slice of “robustness issues” (instability under minor variations) in interactive / multi-turn settings.

## How is this different from GALILEO?

- This paper is a survey/taxonomy; it does not propose a concrete multi-turn protocol, metric suite, or experimental setup focused on dialogue dynamics.
- Scope is much broader than GALILEO: includes informal/fundamental cognitive biases, formal logical reasoning, and embodied reasoning failures.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO provides an operationalized multi-turn evaluation with well-defined perturbations and outcome metrics (e.g., turn-of-failure / time-to-failure style), it can be presented as a *clean instantiation* of the “robustness issues” column in this survey.

## Where GALILEO is weaker / needs to improve

- GALILEO might benefit from mapping its measured failures onto a clearer taxonomy (fundamental vs application-specific vs robustness) and stating which bucket(s) it targets.
- If GALILEO’s task set is narrow, this survey can be used to justify broader coverage or explicit out-of-scope statements.

## Action items for GALILEO (experiments / method / writing)

- [ ] Related work framing: cite this as an umbrella survey for “reasoning failures”, and explicitly place GALILEO under “robustness issues” (instability under minor variations) in multi-turn interaction.
- [ ] Taxonomy mapping: add a short paragraph/figure mapping GALILEO’s settings (e.g., social pressure / stance flips / persuasion) onto the survey’s axes.
- [ ] Terminology alignment: adopt (or contrast with) their “fundamental vs application-specific vs robustness” distinction to clarify what failures GALILEO claims to diagnose.

## Quotes / details to potentially cite

- Abstract (robustness axis): they classify failures into “fundamental failures… application-specific limitations… and robustness issues characterized by inconsistent performance across minor variations.”
- They propose a “2-axis structure (reasoning type × failure type)” taxonomy (see Figure 1 in the paper).
