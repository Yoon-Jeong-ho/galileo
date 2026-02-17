# Sycophancy Claims about Language Models: The Missing Human-in-the-Loop

- Year: 2025
- Venue: NeurIPS 2025 Workshop on LLM Evaluation; ICLR 2025 Workshop on Bi-Directional Human-AI Alignment (arXiv)
- Authors: Jan Batzner, Volker Stocker, Stefan Schmid, Gjergji Kasneci
- URL: https://arxiv.org/abs/2512.00656
- BibTeX key (if we add it): batzner2025sycophancyclaims
- Tags: sycophancy, conceptual, evaluation, human-in-the-loop

## One-sentence takeaway

Most “LLM sycophancy” papers operationalize and measure something like agreeableness/response-alignment, but rarely validate that humans actually perceive the outputs as sycophantic—making cross-paper claims and comparisons shaky.

## What problem does it solve?

- Clarifies a recurring mismatch: **sycophancy is a human-centric concept (insincere flattery to gain favor)**, yet most LLM evaluation work labels behaviors as “sycophancy” without measuring **human perception**.
- Surfaces **conceptual ambiguity** (sycophancy vs personalization vs specification gaming vs agreeableness bias) that undermines comparability across benchmarks.

## What is the core method / protocol?

- This is primarily a **methods/measurement review + taxonomy** paper.
- It groups prior work into **five operationalizations** used to elicit/measure “sycophancy”:
  - persona-based prompts ("I am..." / "You are...")
  - direct questioning / challenge prompts (e.g., "Are you sure?")
  - keyword/query-based manipulation (incl. retrieval contexts)
  - visual misdirection
  - LLM-based evaluations
- Key critique: these operationalizations often **cannot distinguish** (i) true “approval-seeking” behavior from (ii) other effects (prompt sensitivity, politeness norms, personalization, RLHF artifacts), and essentially never test the **human-centric** claim.

## What are the key metrics?

- No new benchmark metric; it critiques that existing metrics are inconsistent and often lack construct validity.
- Emphasizes that to claim “sycophancy”, evaluations should include **human judgments of perceived sycophancy** (or use more precise labels otherwise).

## What are the main results?

- Identifies a **methodological gap**: across the reviewed sycophancy-measurement papers, **none explicitly measures human perception of sycophancy** (even when humans are used, it’s typically for overall quality/performance rather than “sycophancy-ness”).
- Argues for clearer terminology and for using alternatives like **“agreeableness bias”** or **“response alignment”** when human-centric intent/perception is not evaluated.
- Provides concrete recommendations:
  - Terminology: a coherent definition of “AI sycophancy” for cross-study comparability
  - Human-centricity: incorporate human perception measurement
  - Specificity: avoid over-claiming “sycophancy” when measuring narrower behaviors

## How is this similar to GALILEO?

- Directly relevant to **threats-to-validity** for any pressure/agreeability/drift benchmark:
  - it supports a key writing point: “sycophancy” is often a **bucket term**, so we must be explicit about what we measure.
- Motivates GALILEO’s need for **decomposed outcome channels** (e.g., agreement vs hedging vs deference) rather than a single scalar “sycophancy score.”

## How is this different from GALILEO?

- Not an evaluation benchmark or multi-turn protocol; it’s a **conceptual/methodological review**.
- Does not provide:
  - turn-level dynamics (time-to-failure, recovery trajectories)
  - controlled drift-vs-evidence belief revision designs
  - concrete mitigation baselines

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO includes clear operational definitions + controls, we can claim stronger **construct clarity** than much of the prior sycophancy literature.
- Opportunity: explicitly position GALILEO’s measurements as **pressure-driven response alignment / instability** unless and until we add a human-perception layer.

## Where GALILEO is weaker / needs to improve

- If GALILEO claims “sycophancy” without human perception, this paper is a direct critique.
- Even if we avoid the label, we should ensure we don’t implicitly equate:
  - agreement with user preference
  - politeness/validation with “insincere flattery”

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a **human-perception validation slice**: small-scale labeling of outputs for perceived sycophancy vs politeness vs uncertainty/hedging (even 200–500 items can anchor construct validity).
- [ ] In the paper, add a short **terminology box**: distinguish “sycophancy”, “agreeableness bias”, “pressure-driven drift”, and “evidence-driven revision”.
- [ ] When reporting results, avoid over-general claims; phrase as “pressure-induced agreement/instability” unless validated.

## Quotes / details to potentially cite

- "Despite sycophancy being inherently human-centric, current research does not evaluate human perception." (Abstract)
- "...none of those we reviewed explicitly measures how humans perceive sycophantic language model behavior." (Conclusion / Recommendations)
- Suggested terminology shift when human perception is absent: “agreeableness bias” / “response alignment”.
