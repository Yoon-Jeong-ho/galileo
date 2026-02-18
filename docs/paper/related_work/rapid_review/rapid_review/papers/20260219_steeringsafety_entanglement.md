# SteeringSafety: A Systematic Safety Evaluation Framework of Representation Steering in LLMs

- Year: 2025
- Venue: arXiv
- Authors: Vincent Siu; Nicholas Crispino; David Park; Nathan W. Henry; Zhun Wang; Yang Liu; Dawn Song; Chenguang Wang
- URL: https://arxiv.org/abs/2509.13450
- BibTeX key (if we add it): steeringsafety2025siu
- Tags: representation-steering, safety-eval, entanglement, benchmarks, jailbreak, hallucination, bias

## One-sentence takeaway

SteeringSafety is a unified benchmark+codebase to evaluate representation-steering methods across 7 safety “perspectives” and quantify cross-perspective entanglement/side-effects.

## What problem does it solve?

- Prior work on representation steering (activation addition / direction steering) often evaluates a single target behavior, leaving safety side-effects under-measured.
- Safety interventions can be *entangled*: improving one axis (e.g., refusal/jailbreak resistance) can degrade others (e.g., normative judgments, social behavior, political bias).
- Need a systematic, multi-axis evaluation harness to compare steering methods across models and objectives.

## What is the core method / protocol?

- Define a suite of **7 safety perspectives** (with sub-perspectives; overall 17 datasets), including:
  - bias, harmfulness, hallucination, social behaviors, reasoning, epistemic integrity, normative judgment.
- Provide a **modularized framework** that implements multiple steering methods in a unified way:
  - DIM, ACE, CAA, PCA, LAT (plus enhancements like conditional steering).
- Evaluation protocol: steer toward a target perspective, then evaluate performance across *all* perspectives to measure:
  - steering effectiveness on target
  - collateral changes elsewhere (“entanglement”)
- Uses a mixture of automatic scoring for some datasets (e.g., safety classifiers / LLM judges; details depend on perspective).

## What are the key metrics?

- Target-task success per perspective (varies by dataset: accuracy on MC/boolean tasks; generation safety/factuality rates; etc.).
- Cross-perspective degradation / change relative to baseline (their “entanglement” measurement): how much non-target perspectives shift when steering.

## What are the main results?

- Steering quality is highly dependent on the **(method, model, perspective)** pairing.
- DIM is reported as comparatively consistent, but **all** methods show substantial entanglement.
- Notable reported entanglements:
  - Social behaviors can be highly vulnerable (degradation up to **76%** reported).
  - Jailbreaking-oriented steering can compromise **normative judgment**.
  - Hallucination steering can unpredictably shift **political views**.

## How is this similar to GALILEO?

- Shares the core concern that “safety is multi-dimensional” and interventions can introduce unintended failures.
- Provides a concrete structure (axes, datasets, modular evaluation) that maps well to a GALILEO-style holistic evaluation story.

## How is this different from GALILEO?

- Focuses specifically on **representation steering** interventions (activation-space methods), not general training/alignment methods.
- Emphasizes benchmark construction and comparative evaluation rather than proposing a new safety/control method (as far as the paper sections read here).

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO offers clearer causal mechanisms or more principled guarantees for avoiding cross-attribute leakage, that would be a differentiator vs. SteeringSafety’s empirical “measure and report entanglement” approach.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently evaluates only a narrow slice of safety behaviors, SteeringSafety highlights the risk: strong gains on one axis can mask regressions elsewhere.
- Consider adopting their “perspectives × sub-perspectives” framing for reporting.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add an “entanglement” section: when optimizing/steering for X, report deltas on Y1..Yk (even if only a smaller set than 7 perspectives).
- [ ] If GALILEO uses any steering/conditioning, test for side-effects on social-behavior style benchmarks (where this paper reports biggest vulnerability).
- [ ] In related work, cite SteeringSafety as evidence that representation-steering effects are highly configuration-dependent and require holistic evaluation.

## Quotes / details to potentially cite

- “We introduce SteeringSafety, a systematic framework for evaluating representation steering methods across seven safety perspectives spanning 17 datasets.”
- “Results … reveal that strong steering performance depends critically on pairing of method, model, and specific perspective.”
- “All methods exhibit substantial entanglement: social behaviors show highest vulnerability (… degradation as high as 76%), jailbreaking often compromises normative judgment, and hallucination steering unpredictably shifts political views.”
