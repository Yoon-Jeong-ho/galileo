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

6) **SycoEval-EM: Sycophancy Evaluation of Large Language Models in Simulated Clinical Encounters for Emergency Care** (Wang et al., arXiv 2026)
   - Contributes: a crisp **multi-turn adversarial persuasion** protocol in an applied “guideline adherence” setting; large cross-model sweep (20 models, 1,875 dialogues) with a simple, decision-relevant outcome (**acquiescence to unindicated care**) under multiple persuasion tactics.
   - Misses vs GALILEO: primarily an **endpoint** metric; does not explicitly separate *pressure-driven drift* from *evidence-driven revision*, and does not report a turn-of-failure / recovery trajectory.
   - Borrow: (i) the **patient-pressure tactics** menu, (ii) scenario-dependent vulnerability analysis (CT vs antibiotics vs opioids), (iii) multi-judge majority-vote evaluation to reduce single-judge bias.
   - How to change GALILEO: add an applied “pressure vs guidelines” slice and report both endpoint failures and time-to-failure (first acquiescence turn) plus recovery-after-refusal dynamics.

