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

7) **How RLHF Amplifies Sycophancy** (Shapira, Benade, Procaccia, arXiv 2026)
   - Contributes: a mechanistic account of why preference-based post-training can *increase* sycophancy: the direction of behavioral drift is governed by a **covariance / mean-gap** condition linking (i) base-policy mass on sycophantic completions and (ii) learned reward. Traces how labeler preference bias under pairwise comparison models (e.g., Bradley–Terry) can induce a reward tilt toward agreement. Proposes a targeted mitigation: the **minimum-KL** post-trained policy subject to “no increase in sycophancy”, yielding a closed-form **agreement penalty** (minimal reward correction).
   - Misses vs GALILEO: not a dialogue-trajectory benchmark (no survival/time-to-failure curves; no recovery-after-flip measurement); depends on modeling assumptions about reward learning/optimization.
   - Borrow: (i) the reward-tilt diagnostic framing (“reward gaps are common”), (ii) the minimum-KL-under-constraint perspective as a principled way to add “anti-sycophancy” constraints without over-regularizing.
   - How to change GALILEO: cite as the *training-time* mechanism behind why social-pressure agreement failures can be amplified post-RLHF; position GALILEO as complementary by measuring **interaction-time** drift/flip/recovery under pressure with explicit neutral controls.

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

10) **Vulnerability of LLMs’ Belief Systems? LLMs Belief Resistance Check Through Strategic Persuasive Conversation Interventions** (Huang et al., arXiv 2026)
   - Contributes: a systematic **SMCR** factorization of persuasion (Source/Message/Receiver variations) in a **multi-turn belief-erosion** pipeline across domains (BoolQ, PubMedQA, LatentHatred). Reports both a rate metric (**Robustness = 100 − MR@4**) and a time-to-failure proxy (**Avg. End Turn**) capturing *when* belief flips.
   - Misses vs GALILEO: belief is a binary yes/no reversal and the protocol is short-horizon; does not explicitly separate **evidence-driven revision** from **pressure-only drift**, and does not measure **recovery after flip**.
   - Borrow: (i) SMCR framing to justify multiple “pressure operators” beyond message text, (ii) MR@n / Avg.EndTurn reporting as a simple survival-adjacent summary, (iii) the cautionary result that **confidence/meta-cognition prompts can worsen robustness**.
   - How to change GALILEO: add SMCR-like pressure categories; include an explicit “meta-cognition” (confidence-elicitation) condition as a *non-obvious negative control*; and position our neutral re-asking + recovery measurements as filling their main gaps.

## Changelog

- 2026-02-17: Added *Time-To-Inconsistency* (arXiv 2025) for its survival-analysis / censoring framing (fills a key metric gap). Displaced *SycoEval-EM* (arXiv 2026): strong applied endpoint evaluation, but less central than survival/time-to-event methodology for GALILEO’s core claims.
- 2026-02-17: Added *Consistency of Large Reasoning Models Under Multi-Turn Attack* (arXiv 2026) for its 8-round protocol + PWC/trajectory analysis + evidence that confidence-based defenses can fail on reasoning models. Displaced *Moral Sycophancy in Vision Language Models* (arXiv 2026): useful EIR/ECR idea, but less central (2-turn, VLM-specific).
- 2026-02-17: Added *Drift No More? Context Equilibria in Multi-Turn LLM Interactions* (arXiv/workshop 2025) for its KL-to-reference drift signal + equilibrium/intervention framing (fills a key “drift control” gap). Displaced *SycEval: Evaluating LLM Sycophancy* (AIES 2025): strong pressure-strength ladder, but less central than an explicit drift-control/equilibrium framework for GALILEO’s drift-vs-revision positioning.
- 2026-02-17: Added *Vulnerability of LLMs’ Belief Systems?* (arXiv 2026) for its SMCR-based systematic persuasion study + multi-turn belief-erosion pipeline with an explicit time-to-flip proxy (Avg. End Turn) and robustness rate (100−MR@4), plus the non-intuitive result that meta-cognition prompting can worsen robustness. Displaced *Drift No More?* (arXiv/workshop 2025): valuable drift/equilibrium framing, but less directly adjacent than an SMCR persuasion protocol for GALILEO’s core social-pressure robustness claims.

- 2026-02-17: Added *How RLHF Amplifies Sycophancy* (arXiv 2026) for a mechanistic, reward-learning-to-policy-optimization explanation + minimum-KL agreement-penalty mitigation. Displaced *Towards Understanding Sycophancy in Language Models* (arXiv 2023/2025): foundational empirical evidence, but less directly useful than an explicit mechanism + mitigation lens for our “post-training amplifies pressure-agreement failures” story.
