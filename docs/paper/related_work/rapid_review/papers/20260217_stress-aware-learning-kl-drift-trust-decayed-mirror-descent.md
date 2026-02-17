# Stress-Aware Learning under KL Drift via Trust-Decayed Mirror Descent

- Year: 2025
- Venue: arXiv
- Authors: Gabriel Nixon Raj
- URL: https://arxiv.org/abs/2510.15222
- BibTeX key (if we add it): nixonraj2025stressaware
- Tags: drift, kl, belief-update, mirror-descent, dynamic-regret, robustness

## One-sentence takeaway

A theoretical framework for sequential learning under distribution drift that uses “trust-decayed” (stress-aware, exponentially tilted) mirror descent to achieve ~O(\sqrt{T}) dynamic regret under a KL-drift path-length measure.

## What problem does it solve?

- Online/sequential decision-making when the data distribution changes over time (non-stationarity), where standard Bayesian updates / mirror descent “over-trust” stale information and suffer large losses after regime changes.
- Provides robustness notions and guarantees that explicitly track drift measured via KL(D_t || D_{t-1}).

## What is the core method / protocol?

- Introduces an entropy-regularized **trust-decay** mechanism that applies **exponential tilting** (“stress-aware” reweighting) to:
  - belief updates, and
  - mirror-descent decisions.
- Shows (on the simplex) a **Fenchel-dual equivalence** between belief tilt and decision tilt.
- Uses a KL-drift **path length** S_T = \sum_{t\ge 2} \sqrt{ KL(D_t || D_{t-1}) / 2 } to characterize non-stationarity.
- Provides a parameter-free hedge variant to adapt tilt intensity to unknown drift; warns that persistent over-tilting incurs a stationary-regime penalty.

## What are the key metrics?

- **Dynamic regret** vs a time-varying comparator: \n  Reg_T = \sum_t f_t(x_t) - \sum_t f_t(x_t^*)
- Drift measure: per-round KL, and cumulative path length S_T.
- Robustness notions introduced:
  - **Fragility**: worst-case excess risk over a KL ball around a reference distribution.
  - **Belief bandwidth**: largest KL radius tolerable before excess loss exceeds a threshold.
  - **Decision-space Fragility Index** (as summarized in abstract).

## What are the main results?

- High-probability sensitivity bounds relating KL drift to changes in expected loss.
- Dynamic-regret guarantees of \tilde{O}(\sqrt{T}) under KL-drift path length S_T.
- Claim: trust-decay achieves **O(1) per-switch regret**, while “stress-free” updates have \Omega(1) tails (worse switching behavior).
- Over-tilting: \Omega(\lambda^2 T) penalty in stationary environments.

## How is this similar to GALILEO?

- Conceptual overlap: both care about **drift** and **robustness under distribution shift**.
- Provides a clean *mathematical drift budget* (KL path length) and robustness diagnostics (fragility/bandwidth) that could inspire:
  - drift-aware evaluation metrics,
  - “trust decay” or reweighting of evidence/history during multi-turn interaction.

## How is this different from GALILEO?

- This is theory-heavy online convex optimization / sequential decision-making; not an LLM dialogue evaluation paper.
- Uses KL drift between data distributions as the primitive; GALILEO’s drift may be interaction-induced (prompting/user pressure) and not naturally expressible as KL(D_t||D_{t-1}) without a modeling choice.

## Where GALILEO is stronger / cleaner (if true)

- GALILEO can directly operationalize drift in **multi-turn LLM behaviors** with concrete tasks/benchmarks, whereas this work is general theory and may be hard to instantiate for LLM conversations without strong assumptions.

## Where GALILEO is weaker / needs to improve

- GALILEO could benefit from a more explicit **drift budget** / variation measure (analogous to S_T) and from robustness notions like fragility/bandwidth to ground evaluation and claims.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider defining a “conversation drift path length” proxy (not necessarily KL) and showing performance/robustness scaling with it, analogous to S_T.
- [ ] Add a short related-work paragraph mapping “trust decay / exponential tilting” to mechanisms like downweighting stale context, adaptive skepticism, or stress-aware reweighting in multi-turn evaluation.
- [ ] Consider a diagnostic inspired by **fragility**: worst-case degradation under bounded “pressure” perturbations (bounded intervention set) around a baseline interaction.

## Quotes / details to potentially cite

- Abstract: dynamic-regret guarantees of \tilde{O}(\sqrt{T}) under KL-drift path length S_T = \sum_{t\ge2}\sqrt{\mathrm{KL}(D_t\|D_{t-1})/2}.
- Abstract: belief tilt and decision tilt coincide (Fenchel-dual equivalence) on the simplex.
- Abstract: over-tilting yields \Omega(\lambda^2 T) stationary penalty.
