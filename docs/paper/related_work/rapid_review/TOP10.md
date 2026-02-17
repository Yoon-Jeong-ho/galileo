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

3) **Sycophancy Hides Linearly in the Attention Heads** (Genadi et al., arXiv 2026)
   - Contributes: a mechanistic localization claim that **correct→incorrect sycophancy** is most *steerable* via a **sparse subset of middle-layer attention heads**, plus evidence those heads attend disproportionately to **user doubt** cues.
   - Misses vs GALILEO: intervention relies on **internal activation access** + per-model head selection; less emphasis on long-horizon survival/recovery metrics (focus is “where to steer” more than trajectory evaluation).
   - Borrow: (i) the “**separable vs steerable**” distinction (a useful framing for why some internal sites matter), (ii) head-level sparsity as a hypothesis about where social-pressure cues route, (iii) probe transfer tests across factual QA benchmarks.
   - How to change GALILEO: add an interpretability appendix: test whether our pressure-driven flips are linearly decodable and whether a small set of attention heads dominates (and whether those heads track doubt/uncertainty tokens), to strengthen causal/diagnostic narrative.

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

7) **Modeling and Predicting Multi-Turn Answer Instability in Large Language Models** (He et al., arXiv 2025)
   - Contributes: a very simple but strong multi-turn *re-questioning* protocol (“Think again” / “Are you sure?” / “You are wrong”) and a principled long-horizon metric: **stationary (long-run) accuracy**, estimated by fitting a **Markov chain** over correctness transitions across turns; also explores **linear probes** to predict forthcoming answer changes.
   - Misses vs GALILEO: limited controls to separate *pressure-only drift* from *evidence-driven correction*; stationary accuracy can hide recovery/oscillation structure; relies on internal hidden states for the probe result (may not be available for closed models).
   - Borrow: (i) the “**stationary accuracy**” framing as an interactive robustness headline metric, (ii) Markov transition reporting as a compact summary of multi-turn dynamics, (iii) the claim that repeated challenges without new evidence are a realistic deployment stressor.
   - How to change GALILEO: add stationary-accuracy (or stationary truth-rate) as an additional reporting lens, and compare it against ToF/survival/PWC + explicit recovery metrics to show what each summary captures/misses.

8) **Persuasion Dynamics in LLMs: Investigating Robustness and Adaptability in Knowledge and Safety with DuET-PD** (Tan et al., EMNLP 2025; arXiv 2025)
   - Contributes: a clean **dual-axis** multi-turn persuasion protocol that separately measures (i) **robustness to misleading persuasion** (NEG) and (ii) **receptiveness to corrective persuasion** (POS), across **knowledge (MMLU-Pro)** and **safety (SALAD-Bench)**. Provides simple, highly reportable metrics (Acc@0, POS/NEG Acc@3, POS/NEG Flip@3) and highlights primacy effects (turn-1 dominates).
   - Misses vs GALILEO: MCQ framing + synthetic persuasion generation; POS is “correction” rather than evidence-bearing belief revision; only 3 turns (no long-horizon survival curves, no explicit recovery-after-flip objective).
   - Borrow: (i) the **NEG vs POS** decomposition (gullibility vs stubbornness) as the central story, (ii) their capability–adaptability trade-off discussion, (iii) their mitigation framing showing resist-only tuning can destroy receptiveness.
   - How to change GALILEO: mirror their dual reporting (pressure-vs-correction) and add an intervention baseline that explicitly optimizes for **both** resist + update (not resist-only).

9) **Consistency of Large Reasoning Models Under Multi-Turn Attack** (Li, Krishnan, Padman, arXiv 2026)
   - Contributes: a direct, **8-round randomized multi-turn attack protocol** on factual MCQ that isolates *answer flipping under pressure* for **frontier reasoning models**, plus strong reporting: Acc_init/Acc_avg, **Position-Weighted Consistency (PWC)**, trajectory-pattern taxonomy (recovery vs oscillation vs terminal capitulation), and an attack-type vulnerability profile.
   - Misses vs GALILEO: does not cleanly separate **evidence-driven revision** from pressure-only drift (attacks are rhetorical/social rather than “new evidence” controls); limited focus on explicit recovery objectives beyond pattern counts.
   - Borrow: (i) PWC as an early-failure-sensitive metric, (ii) randomized follow-up ordering to reduce position bias, (iii) failure-mode labels (Self-Doubt, Social Conformity, Suggestion Hijacking, Emotional Susceptibility, Reasoning Fatigue).
   - How to change GALILEO: include a “misleading suggestion” pressure operator as a canonical strong adversary; add PWC/trajectory-pattern reporting; and explicitly note that **confidence-based defenses (CARG) can fail on reasoning models** due to reasoning-induced overconfidence—so GALILEO should avoid over-claiming logprob confidence as an uncertainty signal.

10) **Sycophancy Is Not One Thing: Causal Separation of Sycophantic Behaviors in LLMs** (Vennemeyer et al., arXiv 2025)
   - Contributes: a clean decomposition of “sycophancy” into **sycophantic agreement (SyA)** vs **sycophantic praise (SyPr)** (and contrasts with **genuine agreement (GA)**), showing they are **distinct linear directions/subspaces** in residual activations and **independently steerable** via activation addition.
   - Misses vs GALILEO: mostly controlled/synthetic + mechanistic; limited multi-turn trajectory metrics (time-to-failure, recovery) and may not transfer directly to black-box settings.
   - Borrow: (i) the insistence that we should not treat “sycophancy” as one scalar, (ii) the GA vs SyA separation framing (avoid conflating “agreement” with “deference”), (iii) the “behavior-selective intervention” argument.
   - How to change GALILEO: add/report separate channels for **stance agreement under incorrect user claims** vs **praise/flattery**, and position GALILEO’s decomposed metrics as necessary because these mechanisms are separable.
## Changelog

- 2026-02-18: Added *Sycophancy Hides Linearly in the Attention Heads* (arXiv 2026) for its head-level localization + linear-probe steering story and attention-to-doubt evidence. Displaced *Sycophancy under Pressure* (arXiv 2025): useful mitigation baseline, but less central than a mechanistic “where does the flip live?” neighbor for GALILEO’s core pressure-driven flip claims.

- 2026-02-17: Added *Sycophancy Is Not One Thing: Causal Separation of Sycophantic Behaviors in LLMs* (arXiv 2025) for its mechanistic evidence that **sycophantic agreement** vs **sycophantic praise** vs **genuine agreement** are separable and independently steerable. Displaced *Persuasion Propagation in LLM Agents* (arXiv 2026): valuable for trace-level agent drift, but less central than clarifying sycophancy’s internal decomposition for GALILEO’s core claims.

- 2026-02-17: Added *Persuasion Propagation in LLM Agents* (arXiv 2026) for its trace-level “persuasion → downstream agent behavior drift” framing (search/source narrowing). Displaced *Firm or Fickle?* (ACL 2025): strong PWC metric/protocol neighbor, but less directly about agentic tool-use traces and belief-state propagation across tasks.

- 2026-02-17: Added *Time-To-Inconsistency* (arXiv 2025) for its survival-analysis / censoring framing (fills a key metric gap). Displaced *SycoEval-EM* (arXiv 2026): strong applied endpoint evaluation, but less central than survival/time-to-event methodology for GALILEO’s core claims.
- 2026-02-17: Added *Consistency of Large Reasoning Models Under Multi-Turn Attack* (arXiv 2026) for its 8-round protocol + PWC/trajectory analysis + evidence that confidence-based defenses can fail on reasoning models. Displaced *Moral Sycophancy in Vision Language Models* (arXiv 2026): useful EIR/ECR idea, but less central (2-turn, VLM-specific).
- 2026-02-17: Added *Drift No More? Context Equilibria in Multi-Turn LLM Interactions* (arXiv/workshop 2025) for its KL-to-reference drift signal + equilibrium/intervention framing (fills a key “drift control” gap). Displaced *SycEval: Evaluating LLM Sycophancy* (AIES 2025): strong pressure-strength ladder, but less central than an explicit drift-control/equilibrium framework for GALILEO’s drift-vs-revision positioning.
- 2026-02-17: Added *Vulnerability of LLMs’ Belief Systems?* (arXiv 2026) for its SMCR-based systematic persuasion study + multi-turn belief-erosion pipeline with an explicit time-to-flip proxy (Avg. End Turn) and robustness rate (100−MR@4), plus the non-intuitive result that meta-cognition prompting can worsen robustness. Displaced *Drift No More?* (arXiv/workshop 2025): valuable drift/equilibrium framing, but less directly adjacent than an SMCR persuasion protocol for GALILEO’s core social-pressure robustness claims.

- 2026-02-17: Added *How RLHF Amplifies Sycophancy* (arXiv 2026) for a mechanistic, reward-learning-to-policy-optimization explanation + minimum-KL agreement-penalty mitigation. Displaced *Towards Understanding Sycophancy in Language Models* (arXiv 2023/2025): foundational empirical evidence, but less directly useful than an explicit mechanism + mitigation lens for our “post-training amplifies pressure-agreement failures” story.
- 2026-02-17: Added *Firm or Fickle?* (ACL 2025) for its **Position-Weighted Consistency (PWC)** metric (early-failure-sensitive and recovery-aware) + operator-mix multi-turn protocol + confidence-aware generation baseline (CARG). Displaced *Vulnerability of LLMs’ Belief Systems?* (arXiv 2026): strong SMCR persuasion framing, but less central than a directly metric/protocol neighbor for multi-turn stability reporting.
- 2026-02-17: Added *DuET-PD* (EMNLP 2025; arXiv 2025) for its dual (misleading vs corrective) multi-turn persuasion evaluation + balanced **Holistic DPO** mitigation. Displaced *PARROT* (arXiv 2025): excellent single-turn paired-pressure protocol, but DuET-PD is more central for GALILEO’s multi-turn robustness/adaptability framing.
- 2026-02-17: Added *Modeling and Predicting Multi-Turn Answer Instability in Large Language Models* (arXiv 2025) for its **Markov / stationary-accuracy** robustness metric and simple re-questioning protocol. Displaced *EvolIF* (arXiv 2025): valuable adaptive-horizon instruction-following benchmark, but less central than stationary-accuracy as a direct metric neighbor for GALILEO’s multi-turn stability claims.
