# Time-To-Inconsistency: A Survival Analysis of Large Language Model Robustness to Adversarial Attacks

- Slug: time-to-inconsistency
- Year: 2025
- Venue: arXiv
- Authors: Yubo Li, Ramayya Krishnan, Rema Padman
- Links:
  - paper: https://arxiv.org/abs/2510.02712
  - html: https://arxiv.org/html/2510.02712v4
  - code (if any): (not found in skim)
- Bibtex: https://doi.org/10.48550/arXiv.2510.02712

## 1) What problem does it study?
How to evaluate and predict **multi-turn conversational robustness** of LLMs under adversarial follow-ups, focusing on **when** a model first becomes inconsistent (time-to-failure) rather than only whether it eventually fails.

Core move: treat “first inconsistent answer” as a **time-to-event** outcome and apply **survival analysis**.

## 2) Experimental setup (what is being measured?)
- Task(s): MT-Consistency benchmark (multi-turn adversarial consistency probing over QA-style questions).
- Perturbation/pressure type: 8 adversarial follow-up prompt patterns: Closed-ended, Open-ended, Misleading, Emotional Appeal, Impolite Tone, Expert Appeal, Consensus Appeal, False Agreement.
- Multi-turn? Y (max horizon H=8 turns). Conversations that stay correct through turn 8 are treated as **right-censored**.
- Metrics:
  - **Event time** T: first turn where answer becomes inconsistent/incorrect (given initially correct).
  - Survival analysis outputs: survival curves S(t), hazards h(t).
  - Predictive eval: **Harrell’s C-index** (discrimination) and **Integrated Brier Score (IBS)** (calibration/overall).
  - Drift covariates: prompt-to-prompt drift, context-to-prompt drift, cumulative drift (cosine distance in sentence-transformer embedding space), plus length/domain/difficulty/model-id.

Data scale: 36,951 turns from 9 LLMs (after filtering to initially correct conversations).

## 3) Key findings (bullet)
- **AFT survival models outperform Cox and RSF** on predicting time-to-inconsistency.
  - Best discrimination reported: **C-index ≈ 0.874** (Weibull / Log-Logistic AFT).
  - Best calibration: **IBS ≈ 0.175** (Weibull AFT + interactions), vs Cox IBS ≈ 0.34.
- **Abrupt semantic drift is a dominant failure driver**:
  - Prompt-to-prompt drift sharply increases hazard; they report e.g. GPT-4o hazard ratio **≈ 4.7** and AFT acceleration factor **≈ 0.15** (large compression of survival time).
- **Cumulative drift is counterintuitively “protective”**:
  - Across models, AFT suggests cumulative drift expands time-to-failure by **~1.4× to 2.6×**, interpreted as “adaptation” among conversations that survive early shocks.
- **Proportional hazards assumption violations matter**:
  - Schoenfeld diagnostics show key drift covariates (esp. p2p drift) violate PH, explaining why Cox can be miscalibrated here.
- **Operational monitoring** (offline): a lightweight AFT-based risk monitor can warn ahead of failure.
  - Alerts before failure for **~76%** of failing conversations, with median lead time **~2 turns**.
  - Modest false alerts: **~19%** of censored (non-failing within horizon) conversations alerted.

## 4) Limitations / threats
- Single benchmark family: all results are on **MT-Consistency** and its particular prompt templates; horizon is only **8 turns**.
- Failure event is **binary “first inconsistency”**; does not distinguish error types (sycophancy vs hallucination vs misunderstanding).
- Drift features depend on a single embedding model and cosine distances; interpretability is limited.
- Monitoring experiment is **retrospective/offline** (no interventions tested).

## 5) How it relates to GALILEO
- What we can cite it for:
  - Strong precedent for treating multi-turn degradation as **time-to-event** and using **survival analysis** instead of (only) flip rates.
  - Shows value of (i) **censoring** and (ii) calibration-aware evaluation (IBS) for trajectory metrics.
  - Provides a concrete “risk monitor” framing: turn-level warning signals based on survival models.
- Where we differ (our delta):
  - GALILEO is centered on **social pressure / persuasion / sycophancy** and needs explicit **drift vs belief-revision controls** and **recovery-after-flip** trajectories; this paper is “inconsistency under adversarial attacks” with drift covariates and does not separate normative revision from pressure drift.
- Direct mapping:
  - Survival ↔ time-to-inconsistency survival curves / hazard modeling (directly adoptable).
  - TOF ↔ their event time T (first inconsistency turn); essentially TOF with censoring.
  - Recovery ↔ not measured (event stops at first inconsistency).
  - Neutral Re-asking Control ↔ not present; they rely on MT-Consistency setup + drift covariates.

## 6) Quote-able lines
- “By reframing multi-turn conversational failure as a time-to-event process…” (abstract / conclusion framing)
- Key conceptual contrast: static aggregate scores cannot distinguish “fails immediately” vs “fails after many turns.”

## 7) Actions
- [ ] Add to paper: related-work section on **time-to-failure metrics** / survival-analysis framing (1 sentence: survival analysis with censoring + hazard as turn-level risk).
- [ ] Add to bib
