# BASIL: Bayesian Assessment of Sycophancy in LLMs

- Year: 2025
- Venue: arXiv (v4 Jan 2026; ACM-style draft)
- Authors: Katherine Atwell; Pedram Heydari; Anthony Sicilia; Malihe Alikhani
- URL: https://arxiv.org/abs/2508.16846
- BibTeX key (if we add it): 
- Tags: sycophancy, bayesian, belief-updating, calibration, evaluation, label-free

## One-sentence takeaway

BASIL proposes a Bayesian decision-theoretic framework that *separates user-driven “sycophantic” belief shifts from rational evidence updating* and quantifies both the magnitude of sycophancy and its deviation from Bayesian-consistent updating—without requiring ground-truth labels.

## What problem does it solve?

- Standard sycophancy evaluations confound:
  - legitimate belief updating when a user statement is *informative evidence*, vs
  - “people-pleasing” conformity to the user as a *social source*.
- Many benchmarks rely on objective ground truth (accuracy flips), which breaks for subjective/uncertain domains where sycophancy is arguably most harmful.

## What is the core method / protocol?

- Elicit model probabilities for a binary outcome and evidence terms (priors, likelihoods, posteriors):
  - prior \(\hat P(X)\)
  - likelihoods \(\hat P(E\mid X)\), \(\hat P(E\mid \neg X)\)
  - posterior \(\hat P(X\mid E)\)
- Compute the model-implied **Bayesian-rational posterior** \(P^*(X\mid E)\) using Bayes’ rule with the model’s own elicited terms.
- Three prompt conditions to isolate “user effect”:
  1) **Abstract**: no outside beliefs stated (baseline).
  2) **Third-party belief**: a third party predicts X (controls for informational/social-proof effect).
  3) **User belief**: “I predict X” (explicit sycophancy probe).
- Metrics:
  - **Descriptive sycophancy**: log-odds change in posterior between conditions (e.g., Third-party → User).
  - **Normative sycophancy impact**: change in Bayesian error (RMSE of \(\hat P(X\mid E)\) vs \(P^*(X\mid E)\)) due to the sycophancy probe.
- Evaluated on 3 uncertainty-driven tasks (as described in the paper): conversation forecasting, morality judgments, cultural acceptability judgments.
- Mitigations:
  - Post-hoc calibration: isotonic regression for priors + **odds-ratio scaling** to propagate calibration through posteriors.
  - Fine-tuning that rewards Bayesian consistency: **BayesSFT** and **BayesDPO** (label-free preference signal = closer to Bayesian-consistent posterior).

## What are the key metrics?

- Log-odds change (LOC) in \(\hat P(X\mid E)\) across conditions (descriptive sycophancy).
- Bayesian-consistency error: RMSE (and KL in appendix) between \(\hat P(X\mid E)\) and \(P^*(X\mid E)\).
- \(\Delta\)RMSE between baseline vs sycophancy-probed condition (normative “sycophancy tax”).

## What are the main results?

- Strong evidence that **user-belief prompts shift posteriors** more than third-party belief prompts (user has an outsized effect even after controlling for “belief as evidence”).
- The *normative* impact depends on the model’s baseline updating tendency:
  - If the model **over-updates**, sycophancy tends to **increase Bayesian error**.
  - If the model **under-updates**, sycophancy can *appear* to reduce error (“right for the wrong reason” compensatory distortion).
- Post-hoc calibration (priors + odds-ratio-scaled posteriors) reduces Bayesian inconsistency; calibrating priors alone can worsen it.
- BayesSFT/BayesDPO reduce Bayesian error and can reduce measured sycophancy.

## How is this similar to GALILEO?

- Directly targets the key conceptual distinction GALILEO cares about: **pressure-driven drift vs evidence-driven revision**.
- Provides a concrete *paired-control* design (Abstract vs Third-party vs User) that is compatible with black-box model evaluation.
- Emphasizes trajectory changes under social pressure and proposes metrics beyond raw flip-rate/accuracy.

## How is this different from GALILEO?

- BASIL is primarily about **probabilistic belief elicitation + Bayesian coherence** (single-step posterior comparisons), not long-horizon multi-turn survival/recovery dynamics.
- Requires eliciting several probability quantities (prior/likelihood/posterior), which may be brittle across prompting styles and models.
- Uses “Bayesian consistency with the model’s own elicited beliefs” as the normative target, whereas GALILEO may want targets tied to external task success or human-grounded outcomes.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO measures **multi-turn time-to-failure and recovery**, it can capture dynamics BASIL largely abstracts away.
- GALILEO can remain closer to *behavioral outcomes* (e.g., maintain-correctness-under-pressure) without requiring probability elicitation plumbing.

## Where GALILEO is weaker / needs to improve

- GALILEO should be explicit about **control conditions** that separate “user statement as evidence” from “user as social pressure”; BASIL offers a clean template.
- If GALILEO currently relies on ground truth, BASIL shows how to build **label-free normative metrics** (internal consistency standards) for subjective domains.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a **Third-party belief** control condition (or analogous “non-user source” condition) to quantify a user-specific conformity residual.
- [ ] Consider a “label-free” **coherence metric**: compare model’s stated update to an internally implied update computed from separately elicited components (not necessarily Bayes; could be GALILEO-specific consistency).
- [ ] In the paper framing, adopt BASIL’s language: *descriptive sycophancy* (magnitude) vs *normative impact* (how it harms rational updating).
- [ ] If we do calibration, avoid “prior-only” calibration; propagate calibration consistently through downstream quantities.

## Quotes / details to potentially cite

- “A central difficulty … is disentangling sycophantic belief shifts from rational changes in behavior driven by new evidence or user-provided information.”
- Three settings for isolation: **Abstract**, **Third-Party**, **User** beliefs; define sycophancy as residual after controlling for evidence/social-proof.
- Descriptive metric uses **log-odds change**; normative metric uses **change in RMSE** to the Bayesian-consistent posterior.
