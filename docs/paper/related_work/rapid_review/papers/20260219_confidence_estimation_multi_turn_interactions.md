# Confidence Estimation for LLMs in Multi-turn Interactions

- Year: 2026
- Venue: arXiv
- Authors: Caiqi Zhang; Ruihan Yang; Xiaochen Zhu; Chengzu Li; Tiancheng Hu; Yijiang Dong; Deqing Yang; Nigel Collier
- URL: https://arxiv.org/html/2601.02179v1
- BibTeX key (if we add it): zhang2026confidence_multiturn (suggested)
- Tags: calibration, confidence, multi-turn, monotonicity, hallucination, agents

## One-sentence takeaway

Introduces a first systematic framework + metrics for *multi-turn* confidence estimation (per-turn calibration + monotonicity under progressively revealed information), showing common confidence heuristics fail and proposing a stronger logit-probe (P(Sufficient)).

## What problem does it solve?

- Existing LLM confidence / uncertainty estimation work is mostly single-turn; it is unclear whether confidence signals behave sensibly when information is revealed across turns.
- For agentic / human-in-the-loop systems, confidence needs to be (a) calibrated at each turn and (b) increase as ambiguity is resolved, otherwise it is not a usable control signal for “ask vs act”.

## What is the core method / protocol?

- Define a controlled multi-turn interaction where *task-relevant information accumulates* each turn.
- Evaluate confidence estimation methods against two desiderata:
  - **Per-turn calibration**: confidence should match empirical accuracy at each turn.
  - **Monotonicity**: confidence should increase as more useful information is provided.
- Contribute datasets / testbeds:
  - **“Hinter–Guesser”** paradigm: under-specified initial query, then progressively revealed clues (“hints”) prune the hypothesis space.
  - Adapt **incremental QA** settings (papers cited include TRICK / GRACE-style hinting benchmarks) for fully-specified-but-hard questions with sequential hints.
- Propose / test a logit-based probe **P(Sufficient)** intended to detect whether the model has enough information to answer reliably (relative to baseline confidence heuristics).

## What are the key metrics?

- **InfoECE**: length-normalized Expected Calibration Error “at information level” to compare calibration across dialogs with different numbers of turns.
- **Kendall’s tau (τ)** to quantify monotonicity of confidence across turns.
- (Also analyze effects of “turn count” vs “added information” on confidence.)

## What are the main results?

- Widely used confidence estimation techniques struggle in multi-turn settings:
  - poor per-turn calibration,
  - confidence often fails to increase monotonically (or increases for the wrong reasons).
- **P(Sufficient)** improves the joint calibration+monotonicity picture relative to baselines, but multi-turn confidence remains “far from solved”.
- Confidence behaves differently in interactive multi-turn vs an equivalent single-turn summary, even when accuracy is similar → the *interaction structure* matters for confidence behavior.

## How is this similar to GALILEO?

- Same meta-problem: **reliability signals across turns** in multi-turn interactions (GALILEO: robustness / drift / pressure; this paper: confidence signal dynamics).
- Emphasizes that single-turn evaluation can be misleading for multi-turn behavior.
- Introduces monotonicity-style turnwise metrics that resemble “drift” tracking (but applied to confidence rather than answers/beliefs).

## How is this different from GALILEO?

- Focuses on **confidence estimation** (calibration + monotonicity) under *information revelation*, not on adversarial/user pressure, persuasion, or sycophantic conformity.
- Uses controlled hinting / progressive disambiguation rather than explicit “pressure prompts” or value/belief challenges.
- Output of interest is a scalar confidence signal; GALILEO’s central object is (presumably) behavioral robustness / belief stability / resistance to pressure across rounds.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO targets *adversarial pressure + robustness under confrontation*, that is a distinct axis not directly covered here.
- GALILEO likely has more direct “safety / misuse” relevance (sycophancy, persuasion, drift under pressure) vs calibration diagnostics.

## Where GALILEO is weaker / needs to improve

- If GALILEO lacks a **calibration-style reliability metric**, this paper suggests a gap: “turn-by-turn” evaluation should include confidence-like signals (even if derived from behavior, e.g., self-consistency, logit margins, abstention).
- If GALILEO claims stability/robustness in multi-turn settings, reviewers may ask for formal monotonicity/calibration metrics analogous to InfoECE/τ.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a “progressive information” condition (hint ladder) alongside “pressure” conditions to disentangle *information gain* vs *social/adversarial pressure* as drivers of drift.
- [ ] Consider reporting a **turnwise calibration metric** for any GALILEO reliability signal (or justify why none is appropriate).
- [ ] Borrow the **monotonicity framing**: define what should be monotone in GALILEO (e.g., confidence in original answer? probability of staying consistent? refusal threshold?) and measure it with Kendall’s τ.
- [ ] Cite this as evidence that multi-turn is a distinct evaluation regime: single-turn reliability does not transfer.

## Quotes / details to potentially cite

- “This work presents the first systematic study of confidence estimation in multi-turn interactions…”
- Desiderata: “per-turn calibration” and “monotonicity of confidence as more information becomes available.”
- Introduces “InfoECE” (length-normalized ECE) and the “Hinter–Guesser” paradigm.
