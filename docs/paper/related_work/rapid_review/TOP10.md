# Rapid Review — Top 10 shortlist (rolling)

This is a rolling shortlist of the **10 most important adjacent papers**.

For each item, we maintain:
- What it contributes
- What it misses relative to GALILEO
- What we should borrow
- How we should change GALILEO (experiments/methods/claims)

> NOTE: This list is expected to change frequently until ~30–50 papers are read.

## Shortlist

1) **Measuring Sycophancy of Language Models in Multi-turn Dialogues** (Hong et al., Findings of EMNLP 2025)
   - Contributes: concrete multi-turn social-pressure protocol + two extremely usable metrics: **Turn of Flip (ToF)** (time-to-failure proxy) and **Number of Flips (NoF)** (instability/oscillation).
   - Misses vs GALILEO: does not explicitly separate *evidence-driven belief revision* from *pressure-driven drift*; limited focus on *recovery after flip*.
   - Borrow: ToF/NoF framing and third-person prompting baseline (reported up to 63.8% reduction in debate).
   - How to change GALILEO: map our metrics onto a ToF/NoF-style vocabulary (or add direct analogues) and highlight our additional controls + recovery measurements.

2) **ELEPHANT: Measuring and understanding social sycophancy in LLMs** (Cheng et al., arXiv 2025)
   - Contributes: a theory-grounded expansion from “explicit agreement” → **social sycophancy as face-preservation** (validation, indirectness, framing, moral inconsistency), plus an evaluation showing LLMs preserve user face far more than humans in advice/wrongdoing contexts.
   - Misses vs GALILEO: does not center multi-turn *time-to-failure* / recovery dynamics; relies on human baselines and social-dimension classifiers rather than drift-vs-revision controls.
   - Borrow: the face-preservation taxonomy; the *stance-swap* moral-consistency probe (user adopts either side) as a clean way to reveal side-dependent affirmation.
   - How to change GALILEO: add “face channels” of pressure as first-class metrics/labels (validation/framing adoption), not only explicit belief agreement; include a moral-consistency or stance-swap subtest.

3) **Sycophancy under Pressure: Evaluating and Mitigating Sycophantic Bias via Adversarial Dialogues in Scientific QA** (Zhang et al., arXiv 2025)
   - Contributes: a scientific-QA-grounded pressure protocol (single-turn + multi-turn) and a practical mitigation baseline (**Pressure-Tune**: SFT on synthetic adversarial dialogues with rationales) targeting “misleading/sycophancy resistance”.
   - Misses vs GALILEO: does not foreground survival/time-to-failure metrics or recovery-after-flip dynamics; resistance rates can obscure *when* failure occurs.
   - Borrow: Pressure-Tune as a lightweight, defensible mitigation baseline; the framing of “user-imposed social pressure distorts outputs” in a factual domain.
   - How to change GALILEO: (i) add a Pressure-Tune-style baseline to our evaluations, (ii) include at least one high-stakes factual slice (e.g., scientific QA) or explicitly argue domain generality.

4) **T3: Benchmarking Sycophancy and Skepticism in Causal Judgment** (Chang, arXiv 2026)
   - Contributes: a clean **sensitivity vs specificity** decomposition (Utility vs Safety; Sheep vs Wolves), plus explicit measurement of **calibrated abstention** (Wise Refusal) and **multi-turn flip dynamics** (Good Flip vs Bad Flip) under social + epistemic pressure.
   - Misses vs GALILEO: centered on causal-judgment vignettes rather than generic belief drift/revision; does not deeply study recovery-after-flip trajectories (focus is more diagnosis than intervention).
   - Borrow: (i) reporting in a 2D Utility/Safety space to avoid “safe-by-refusing-everything” and “helpful-by-agreeing-with-everything” regimes, (ii) GoodFlip/BadFlip as an interpretable stability diagnostic, (iii) WRR/FCR-style calibrated refusal metrics.
   - How to change GALILEO: add an explicit endorse-true vs reject-false breakdown, track good/bad flip asymmetry, and treat calibrated uncertainty as a first-class outcome (not just an error mode).

5) **CausalT5K: Diagnosing and Informing Refusal for Trustworthy Causal Reasoning of Skepticism, Sycophancy, Detection-Correction, and Rung Collapse** (Geng et al., arXiv 2026)
   - Contributes: a large (5k+) **diagnostic** benchmark with (i) **trap taxonomy** + **Pearl ladder rung labels**, (ii) **paired neutral vs pressure variants** enabling explicit flip-quality metrics (Bad Flip, Paranoia, Sycophancy Ratio), and (iii) **Utility/Safety** decomposition so “refuse-everything” and “accept-everything” pathologies are visible.
   - Misses vs GALILEO: primarily a causal-reasoning infrastructure; less emphasis on full *recovery-after-flip trajectories* and more on diagnosis/control landscapes.
   - Borrow: (i) pressure-paired evaluation as a first-class benchmark design choice, (ii) 2D reporting (Utility/Safety) and flip-based axes (paranoia vs sycophancy), (iii) judge-dependence framing (“false competence” under weak audit).
   - How to change GALILEO: report neutral-vs-pressure paired outcomes, include flip-quality rates (good/bad) and consider a 2D “helpful vs safe” breakdown; explicitly test judge/auditor strength as an experimental variable.

6) **Time-To-Inconsistency: A Survival Analysis of Large Language Model Robustness to Adversarial Attacks** (Li, Krishnan, Padman, arXiv 2025)
   - Contributes: a very direct **survival-analysis framing** for multi-turn robustness with censoring (8-turn horizon) and rich reporting beyond flip-rate: hazards, survival curves, and calibrated predictive metrics (**C-index**, **Integrated Brier Score**).
   - Misses vs GALILEO: evaluates “inconsistency under adversarial follow-ups” rather than specifically *social pressure / persuasion*; the event is **first inconsistency** (no recovery-after-flip trajectory); no explicit drift-vs-revision control.
   - Borrow: (i) treat failure as **time-to-event** with censoring (strong precedent for our TOF/survival claims), (ii) calibration-aware evaluation (IBS) for trajectory risk models, (iii) the “turn-level risk monitor” concept (warning several turns before failure).
   - How to change GALILEO: add a survival-analysis reporting layer (hazard + censored survival), and optionally convert our metrics into a risk-monitor score to predict imminent flips under pressure.

7) **Towards Understanding Sycophancy in Language Models** (Sharma et al., arXiv 2023; updated 2025)
   - Contributes: a foundational, multi-model demonstration that assistants exhibit sycophancy under several **paired neutral-vs-pressure** prompt operators (biased feedback, “Are you sure?” challenges, user-belief injection, and mimicking false framing), plus evidence that **human preference data and preference models can incentivize matching the user’s views**.
   - Misses vs GALILEO: interactions are mostly **short-horizon (2–3 turns)**; no explicit survival/time-to-failure curves over long dialogues and no recovery-after-flip measurement.
   - Borrow: the “Are you sure?” and belief-injection operators as simple, reusable pressure primitives; and the argument that **preference learning can create systematic incentives for agreeing with user beliefs**.
   - How to change GALILEO: cite as prior evidence for (i) pressure-induced flips, (ii) the need for neutral-vs-pressure controls, and (iii) why reward/preference training may push toward agreement; then position GALILEO as extending this to **trajectory-level** metrics (TOF/survival/recovery) and drift-vs-revision separation.

8) **PARROT: Persuasion and Agreement Robustness Rating of Output Truth** (Çelebi et al., arXiv 2025; under review)
   - Contributes: a very GALILEO-adjacent **paired neutral vs authoritative-false** protocol on MMLU-style MCQ, plus reporting of (i) **accuracy degradation under pressure**, (ii) **follow-rate** (agreement with imposed false answer), and (iii) **logprob-based confidence shift tracking** (calibration degradation / “epistemic collapse”). Also introduces an **8-state behavioral taxonomy** (incl. robust correct, sycophantic agreement, reinforced error, stubborn error, self-correction, etc.).
   - Misses vs GALILEO: not multi-turn (no explicit time-to-failure / survival curves), and “recovery” is only a coarse category rather than a measured trajectory.
   - Borrow: (i) paired-control design as an explicit causal-ish estimator, (ii) confidence-mass shift metrics toward true vs imposed-false answers, (iii) failure-mode taxonomy for richer reporting than flip-rate alone.
   - How to change GALILEO: add a comparable paired-pressure slice (authority templates) and consider reporting confidence-drift (probability mass movement) alongside ToF/survival + recovery.

9) **Consistency of Large Reasoning Models Under Multi-Turn Attack** (Li, Krishnan, Padman, arXiv 2026)
   - Contributes: a direct, **8-round randomized multi-turn attack protocol** on factual MCQ that isolates *answer flipping under pressure* for **frontier reasoning models**, plus strong reporting: Acc_init/Acc_avg, **Position-Weighted Consistency (PWC)**, trajectory-pattern taxonomy (recovery vs oscillation vs terminal capitulation), and an attack-type vulnerability profile.
   - Misses vs GALILEO: does not cleanly separate **evidence-driven revision** from pressure-only drift (attacks are rhetorical/social rather than “new evidence” controls); limited focus on explicit recovery objectives beyond pattern counts.
   - Borrow: (i) PWC as an early-failure-sensitive metric, (ii) randomized follow-up ordering to reduce position bias, (iii) failure-mode labels (Self-Doubt, Social Conformity, Suggestion Hijacking, Emotional Susceptibility, Reasoning Fatigue).
   - How to change GALILEO: include a “misleading suggestion” pressure operator as a canonical strong adversary; add PWC/trajectory-pattern reporting; and explicitly note that **confidence-based defenses (CARG) can fail on reasoning models** due to reasoning-induced overconfidence—so GALILEO should avoid over-claiming logprob confidence as an uncertainty signal.

10) **Drift No More? Context Equilibria in Multi-Turn LLM Interactions** (Dongre et al., arXiv/workshop 2025)
   - Contributes: a trajectory-first drift definition and dynamical framing: **per-turn KL divergence to a goal-consistent reference policy** \(D_t = D_{KL}(q_t\|p_t)\), with empirical evidence that multi-turn drift often reaches **stable, noise-limited equilibria** (not runaway monotone decay), and that simple **reminder interventions** shift the equilibrium downward.
   - Misses vs GALILEO: not about social pressure/persuasion; relies on access to a reference policy’s token distribution; does not explicitly separate evidence-driven belief revision from pressure-driven drift nor measure recovery-after-flip.
   - Borrow: (i) treat drift as an explicit **time series** (not a scalar end score), (ii) “equilibrium + intervention” language as a clean way to motivate control conditions and reminder-style baselines, (iii) derived metrics like “turn until divergence exceeds threshold” as a TOF analogue.
   - How to change GALILEO: add a drift-control subsection that (a) defines an operational drift signal over turns, (b) includes reminder interventions, and (c) reports whether behavior is equilibrium-like vs runaway under different pressures.

## Changelog

- 2026-02-17: Added *Time-To-Inconsistency* (arXiv 2025) for its survival-analysis / censoring framing (fills a key metric gap). Displaced *SycoEval-EM* (arXiv 2026): strong applied endpoint evaluation, but less central than survival/time-to-event methodology for GALILEO’s core claims.
- 2026-02-17: Added *Consistency of Large Reasoning Models Under Multi-Turn Attack* (arXiv 2026) for its 8-round protocol + PWC/trajectory analysis + evidence that confidence-based defenses can fail on reasoning models. Displaced *Moral Sycophancy in Vision Language Models* (arXiv 2026): useful EIR/ECR idea, but less central (2-turn, VLM-specific).
- 2026-02-17: Added *Drift No More? Context Equilibria in Multi-Turn LLM Interactions* (arXiv/workshop 2025) for its KL-to-reference drift signal + equilibrium/intervention framing (fills a key “drift control” gap). Displaced *SycEval: Evaluating LLM Sycophancy* (AIES 2025): strong pressure-strength ladder, but less central than an explicit drift-control/equilibrium framework for GALILEO’s drift-vs-revision positioning.

