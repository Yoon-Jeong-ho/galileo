# Auditing cognitive drift in AI-driven recommendation: a responsible AI methods protocol with a health case demonstration

- Year: 2025
- Venue: Frontiers in Neuroscience
- Authors: (Frontiers article; see paper for full author list)
- URL: https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2025.1697053/full
- BibTeX key (if we add it): auditing_cognitive_drift_frontiers_2025
- Tags: cognitive-drift, auditing, protocol, recommender-systems, governance

## One-sentence takeaway

Defines an **auditable protocol** for “cognitive drift” under recommender exposure paths via an interpretable **Cognitive Drift Index (CDI)** (emotion drift + tag homogeneity + selective salience) and maps the score to **governance action bands**.

## What problem does it solve?

- Slow, cumulative changes in what people attend to / how they evaluate information under algorithmic curation are hard to observe and therefore hard to **audit**.
- Existing narratives (filter bubbles, selective exposure) don’t directly provide a **reproducible measurement protocol** for “drift” attributable to exposure path structure.

## What is the core method / protocol?

- Define “cognitive drift” as a composite index (**CDI**) with three interpretable components:
  - **Emotional drift (EB):** round-to-round change in average valence (smoothed).
  - **Tag homogeneity (TC):** within-round tag similarity (mean pairwise **Jaccard** over tokenized tags).
  - **Selective salience (SS):** concentration of attention across content categories (uses attention shares from clicks + dwell time).
- Two-step workflow:
  1) **Calibration run** to set reasonable ranges/thresholds for each component and for CDI; define **action bands** (lightweight interventions like “why am I seeing this?”, diversity rules, affect guards).
  2) **Path-effect test** on observational data from different exposure-path structures (active search vs semi-recommendation vs pure recommendation), using **Type-II WLS ANOVA**, plus robustness checks (±10% weight tweaks; OLS with HC3 robust SEs).
- Emphasis throughout: interpretability + sensitivity + reproducibility + portability across domains.

## What are the key metrics?

- CDI and its components (EB/TC/SS).
- A decomposition framing (as described): update amplitude, direction/polarity, and conflicting cue weights (conceptual mapping to neurocognitive mechanisms).
- Governance-facing: action-band thresholds derived from calibration.

## What are the main results?

- The protocol can be run end-to-end with basic, auditable signals (metadata + coarse attention proxies) and yields stable patterns under their robustness checks.
- Reporting CDI components alongside the composite helps interpret *what kind* of drift is occurring (affect-driven vs structure-driven / homogeneity / salience concentration).

## How is this similar to GALILEO?

- Shares the core stance that “drift” should be treated as a **measurable, monitorable** phenomenon, not just a narrative.
- Strong precedent for:
  - **Interpretable composite indices** with readable subcomponents.
  - A **calibration → thresholding → governance action** pipeline.
  - Robustness checks that show conclusions are not brittle to small weighting changes.

## How is this different from GALILEO?

- Domain: recommender exposure to short-form health videos (population/path-level), not multi-turn conversational belief dynamics in LLMs.
- Signals: relies on **content metadata + attention proxies** (clicks, dwell time), not conversational trajectories, stance/answer flips, or evidence-vs-pressure controls.
- Does not directly operationalize “drift vs evidence-driven revision” in the way GALILEO targets.

## Where GALILEO is stronger / cleaner (if true)

- GALILEO can offer cleaner causal contrasts for LLM settings: **pressure-only** perturbations vs **evidence-bearing** updates; explicit **recovery** dynamics after a flip.
- GALILEO’s unit of analysis (turn-level trajectory) can enable time-to-event / survival-style reporting that is more precise than aggregate indices.

## Where GALILEO is weaker / needs to improve

- Governance integration: this paper is explicit about mapping measurements to **action bands** (what to do at different drift levels). GALILEO could benefit from similarly explicit “what should an operator do?” thresholds.
- Component interpretability: CDI’s “show the parts next to the whole” reporting is a good template to emulate.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a “**monitoring + governance**” subsection: define intervention bands for GALILEO metrics (e.g., low/med/high pressure-susceptibility) with proportionate mitigations.
- [ ] Consider a **composite index** that combines (i) flip timing (ToF / survival), (ii) oscillation/recovery, and (iii) salience-like concentration (e.g., narrowing of cited evidence/sources or rationale diversity), while still reporting each component.
- [ ] Include a **calibration recipe**: a small run that sets default thresholds and demonstrates portability across tasks.

## Quotes / details to potentially cite

- They define cognitive drift as “the effect that accumulates through long-term, low-frequency contact with information, in which subtle changes in evaluative attention and interpretive focus build up.”
- CDI components (as defined): emotional drift (valence change), tag homogeneity (Jaccard similarity), selective salience (attention concentration across categories).
- Workflow framing: calibration to set bands → path-effect testing → robustness checks (±10% weight tweaks; HC3).
