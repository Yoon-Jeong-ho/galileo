# Machine Learning Robustness: A Primer (Chapter 0)

- Year: 2025
- Venue: arXiv (HTML version; book chapter “Chapter 0”)
- Authors: Houssem Ben Braiek, Foutse Khomh
- URL: https://arxiv.org/html/2404.00897v2
- BibTeX key (if we add it): benbraiek2024mlrobustnessprimer (suggested)
- Tags: robustness, trustworthy-ai, survey, adversarial, distribution-shift, testing

## One-sentence takeaway

A broad primer that defines ML robustness as performance stability under specified input changes (with tolerance), and surveys assessment + improvement techniques spanning adversarial and non-adversarial shifts.

## What problem does it solve?

- Establishes a clear conceptual framing of “robustness” for ML as distinct from (i.i.d.) generalization and positions robustness as a key requirement for trustworthy AI.
- Provides a taxonomy-style overview of threats (data shift, drift, adversarial manipulation) and of evaluation/mitigation techniques.

## What is the core method / protocol?

- Conceptual/survey chapter (not a new algorithm):
  - Defines robustness and refines it to require (i) a specified domain of input changes and (ii) a permitted tolerance level for performance degradation.
  - Discusses robustness assessment avenues (adversarial attacks incl. physical/digital; non-adversarial shifts; DL software testing).
  - Discusses robustness improvement strategies (data-centric, model-centric like adversarial training / transfer learning / randomized smoothing, and post-training like ensembles/pruning/repair).

## What are the key metrics?

- No single metric proposed; emphasizes that robustness must be quantified relative to:
  - a defined perturbation/shift set (“domain of potential changes”), and
  - a task/application-specific tolerance threshold (how much degradation is acceptable).

## What are the main results?

- Not an empirical paper; main “results” are definitional clarifications and a structured overview:
  - Robustness is “necessary but not sufficient” on top of i.i.d. generalization.
  - Robustness is framed as causal: performance stability depends on the specified change domain.
  - Robustness is tied into trustworthy AI requirements (safety/robustness intersects with uncertainty quantification and OOD detection).

## How is this similar to GALILEO?

- If GALILEO is about robustness assurance/evaluation, this chapter provides a useful umbrella framing and vocabulary (robustness vs generalization; adversarial vs non-adversarial; assessment vs improvement).
- Highlights the need to specify the perturbation/shift domain and acceptable tolerance—useful for defining GALILEO’s threat model and evaluation protocol.

## How is this different from GALILEO?

- Survey/primer rather than a concrete, testable system contribution.
- Likely broad (covers many families of methods) rather than prescribing a single pipeline/benchmark.

## Where GALILEO is stronger / cleaner (if true)

- A GALILEO-style contribution can be stronger by offering:
  - a concrete operational definition (exact perturbation set + pass/fail criteria),
  - an end-to-end protocol with reproducible tooling and quantitative results.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently focuses on a narrow robustness slice (e.g., only adversarial or only shift testing), this chapter is a reminder to explicitly scope what is (and is not) covered.
- Ensure GALILEO clearly states tolerance thresholds and perturbation domains rather than using “robust” informally.

## Action items for GALILEO (experiments / method / writing)

- [ ] In the paper intro/related-work, explicitly define robustness in terms of (a) perturbation/shift domain and (b) tolerance level; cite this chapter for the framing.
- [ ] Add a short paragraph contrasting i.i.d. generalization vs robustness (dynamic environments) and motivate why robustness evaluation needs explicit change models.
- [ ] If relevant, mention robustness as part of trustworthy AI and its linkage to uncertainty quantification + OOD detection (positioning).

## Quotes / details to potentially cite

- “ML Model robustness denotes the capacity of a model to sustain stable predictive performance in the face of variations and changes in the input data.”
- Refined deployment-oriented definition: a model is robust if specified input variations “do not degrade the model’s predictive performance below the permitted tolerance level.”
- Robustness complements i.i.d. generalizability; i.i.d. generalization is necessary but not sufficient for robustness.
