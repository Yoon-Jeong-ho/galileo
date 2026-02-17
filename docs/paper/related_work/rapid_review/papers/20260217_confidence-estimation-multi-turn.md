# Confidence Estimation for LLMs in Multi-turn Interactions

- Year: 2026
- Venue: arXiv
- Authors: Caiqi Zhang, Ruihan Yang, Xiaochen Zhu, Chengzu Li, Tiancheng Hu, Yijiang Dong, Deqing Yang, Nigel Collier
- URL: https://arxiv.org/abs/2601.02179
- BibTeX key (if we add it): zhang2026confidence-multiturn
- Tags: multi-turn, confidence-estimation, calibration, monotonicity, uncertainty

## One-sentence takeaway

Multi-turn confidence is *not* well-handled by common confidence heuristics; the paper proposes a turn-normalized calibration metric (InfoECE) and a logit-probe **P(Sufficient)** that better tracks “do we have enough information yet?” than classic **P(True)**.

## What problem does it solve?

- Prior confidence-estimation work is mostly **single-turn**; in real interactions, information accumulates across turns and ambiguity can be resolved gradually.
- In such settings, a confidence signal should be (i) **calibrated per turn** and (ii) **monotone increasing** as more task-relevant information arrives.

## What is the core method / protocol?

- Define a *controlled* multi-turn evaluation where each turn reveals **additional task-relevant information** (progressive disclosure).
- Two regimes:
  - **Under-specified initial query** (many plausible answers initially) with progressively revealed clues.
  - **Fully-specified but difficult** initial query (unique answer exists, but hard until hints accumulate).
- Dataset construction:
  - Introduce a **“Hinter–Guesser”** paradigm to generate fixed dialogues for evaluation (the history is held fixed during evaluation; i.e., no strategic question-asking is evaluated).
  - Under-specified datasets: **20Q**-style entity guessing and **Guess-my-City**.
  - Fully-specified incremental-hint datasets: adapt **TrickMe** and **GRACE** (incremental QA).

## What are the key metrics?

- **InfoECE**: a length-normalized (information-level) expected calibration error.
  - Normalize each turn position by *fraction of dialogue completed* (i / L), bin into “information levels”, then compute average |accuracy − confidence| per bin and average across bins.
- **Monotonicity**: **Kendall’s τ** over turn-indexed confidence within each dialogue (average across dialogues).

## What are the main results?

- Common approaches (verbalized confidence prompting, self-consistency sampling, classic logit probes like **P(True)**) often fail to be simultaneously:
  - well-calibrated at each information level, and
  - monotone as more information is revealed.
- Proposed logit probe **P(Sufficient)** performs *comparatively better* on both desiderata (especially in under-specified settings), but the overall problem remains unsolved.
- Additional analyses (from intro): confidence increases can be driven by **turn count / conversation progression** rather than genuine information gain; **P(Sufficient)** better distinguishes meaningful info from filler.

## How is this similar to GALILEO?

- Same core object: **multi-turn dynamics** of a model property (here: confidence; in GALILEO: belief/stance under pressure, drift vs revision, recovery).
- Directly relevant to how we might:
  - justify a **trajectory-level** view (not single-turn), and
  - add calibration / monotonicity checks for any “risk” or “drift” score we report.

## How is this different from GALILEO?

- Not about **social pressure / persuasion / sycophancy**; the multi-turn structure is “progressive hinting” rather than adversarial social influence.
- Does not propose survival / time-to-flip metrics, nor recovery-after-failure protocols; focus is **confidence correctness** under accumulating evidence.

## Where GALILEO is stronger / cleaner (if true)

- GALILEO’s key novelty (pressure-induced drift vs evidence-driven revision with controls + recovery) remains orthogonal; this work is primarily about **confidence signals**.

## Where GALILEO is weaker / needs to improve

- If GALILEO uses any scalar “risk/uncertainty/drift” signal across turns, we should:
  - test **per-turn calibration** (not just correlation with failure), and
  - test **monotonicity** under conditions where more “evidence” should reduce uncertainty.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding a small appendix experiment: treat a GALILEO run as a multi-turn trajectory and evaluate monotonicity/calibration of any *internal* confidence proxy we might use (e.g., judge score, self-reported uncertainty, or logit probe).
- [ ] If we introduce a “sufficiency” notion (e.g., “have we seen enough evidence to return to truth after a flip?”), cite **P(Sufficient)** as precedent for probing sufficiency rather than correctness.

## Quotes / details to potentially cite

- Core desiderata: **per-turn calibration** + **monotonicity** of confidence as information increases.
- New metric: **InfoECE** (information-level, length-normalized ECE).
- New paradigm: **Hinter–Guesser** to generate controlled multi-turn evaluation dialogues.
- New probe: **P(Sufficient)**: ask whether the *current information is sufficient* to entail the answer is uniquely correct (vs asking whether the produced answer is correct).
