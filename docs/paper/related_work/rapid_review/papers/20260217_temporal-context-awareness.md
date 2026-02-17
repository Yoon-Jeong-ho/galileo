# Temporal Context Awareness: A Defense Framework Against Multi-turn Manipulation Attacks on Large Language Models

- Year: 2025
- Venue: IEEE CAI (per arXiv comments; short paper)
- Authors: Assaf Namer
- URL: https://arxiv.org/abs/2503.15560
- BibTeX key (if we add it): namer2025tca
- Tags: multi-turn, jailbreak, defense, temporal, manipulation

## One-sentence takeaway

Proposes a *temporal / cross-turn* safety defense (TCA) that tracks semantic drift + intent consistency across dialogue and uses progressive risk scoring to flag multi-turn manipulation that single-turn filters miss.

## What problem does it solve?

- Multi-turn manipulation / prompt-injection style attacks that unfold gradually (benign turns that “set up” later harmful requests), evading turn-local safety checks.
- Detecting *subtle*, time-extended intent shifts while preserving normal conversation utility.

## What is the core method / protocol?

- “Temporal Context Awareness (TCA)” framework with three main components:
  - Dynamic context embedding analysis to monitor **semantic drift** across turns.
  - Cross-turn consistency verification to assess **intention consistency** (whether the user’s goal is coherently evolving vs. adversarially steering).
  - Progressive risk scoring that accumulates evidence over time (rather than treating each turn independently).
- Paper positions itself as a defense layer for conversational systems; evaluation described as preliminary and based on simulated adversarial scenarios.

## What are the key metrics?

- Not fully specified in the portions reviewed; the paper emphasizes detection of manipulation patterns via:
  - measured semantic drift over turn embeddings
  - intention-consistency signals
  - an aggregate / progressive risk score
- Mentions success-rate style outcomes (detect vs miss) and references prior work reporting high bypass rates under turn reordering (“Speak Out of Turn”).

## What are the main results?

- Preliminary results suggest TCA can identify subtle multi-turn manipulation patterns that are missed by traditional single-turn detectors.
- The paper is high-level; it does not (in the reviewed sections) provide a large benchmark or extensive quantitative comparison.

## How is this similar to GALILEO?

- Shares the core premise that *multi-turn* evaluation/defense requires tracking *trajectory-level* behavior rather than single-turn outcomes.
- Uses temporal notions (drift / accumulation) that align conceptually with turn-by-turn degradation and “time-to-failure” framing.

## How is this different from GALILEO?

- TCA is a **defense / monitoring** proposal (semantic drift + intent consistency + risk scoring), whereas GALILEO is primarily an **evaluation framework** (measuring robustness / degradation across turns).
- Emphasis is security-focused (prompt injection / manipulation) rather than general multi-turn reliability metrics across tasks.

## Where GALILEO is stronger / cleaner (if true)

- GALILEO can provide clearer, reproducible *evaluation protocols* and metrics (e.g., explicit failure definitions, survival/time-to-failure style summaries) rather than a qualitative “risk score” layer.
- Likely stronger empirical breadth (across tasks/models/conditions) if we position GALILEO as a measurement standard.

## Where GALILEO is weaker / needs to improve

- We may currently under-emphasize *defensive* instrumentation (online detection/mitigation) versus only reporting degradation.
- Opportunity: add a “monitoring baseline” family (drift/risk-score detectors) as a comparison point.

## Action items for GALILEO (experiments / method / writing)

- [ ] Related-work framing: cite this as an example of **context-aware defense** that operationalizes semantic drift + cross-turn consistency.
- [ ] Consider adding an optional “online monitor” baseline: progressive risk scoring over turn-level features (embeddings, policy-violation likelihood, refusal pressure), evaluated with our time-to-failure metrics.
- [ ] Clarify in our paper the distinction between (a) *detecting* drift/manipulation and (b) *measuring* robustness outcomes; position GALILEO in (b) while enabling (a) via logged trajectories.

## Quotes / details to potentially cite

- “multi-turn manipulation attacks … exploit the temporal nature of dialogue to evade single-turn detection methods” (abstract).
- TCA integrates “dynamic context embedding analysis, cross-turn consistency verification, and progressive risk scoring” (abstract).
- Related-work pointer: “Speak Out of Turn” is cited as revealing turn-order exploitation and reporting high bypass success under turn reordering (paper discussion; exact figure should be verified from the cited source).
