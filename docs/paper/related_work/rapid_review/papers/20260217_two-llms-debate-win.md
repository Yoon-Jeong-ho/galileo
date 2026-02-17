# When Two LLMs Debate, Both Think They’ll Win

- Year: 2025
- Venue: arXiv (cs.CL)
- Authors: Pradyumna Shyama Prasad; Minh Nhat Nguyen
- URL: https://arxiv.org/abs/2505.19184
- BibTeX key (if we add it): PrasadNguyen2025TwoLLMsDebate
- Tags: multi-turn, debate, calibration, overconfidence, belief-updating

## One-sentence takeaway

In a controlled zero-sum, multi-turn debate setup, frontier LLMs show systematic and *escalating* overconfidence (often mutually impossible), suggesting weak metacognitive belief/confidence updating under opposition.

## What problem does it solve?

- Measures whether LLMs can **revise confidence** appropriately in a **dynamic, adversarial, multi-turn** setting (vs. static calibration on QA).
- Creates a setting where **mutual overconfidence is formally detectable** via a **zero-sum** structure (win probabilities should roughly sum to 100%).

## What is the core method / protocol?

- Simulated **policy debates**:
  - 60 debates total, each **3 rounds**.
  - 10 “state-of-the-art” LLMs in a debater pool.
  - Motions: 6 policy motions (topics listed in appendix per paper).
- After each round, each debater produces:
  - a **private numeric confidence** (0–100) for “probability I win”, and
  - private text reasoning (scratchpad-style) for the confidence.
- Key design choice: **zero-sum framing** (one side “wins”), so if both sides claim high win probability simultaneously, this implies miscalibration / self-bias.
- Ablations discussed in the paper (per abstract/outline): **self-debate** (model debates an identical copy) and “informed self-debate” (told win chance is exactly 50%).

## What are the key metrics?

- Primary:
  - **Initial confidence** (before seeing opponent arguments).
  - **Confidence trajectory** across rounds (opening → rebuttal → closing).
  - **Mutual high-confidence rate**: fraction of debates where both sides report ≥75%.
- Additional analysis:
  - Alignment between **private reasoning** and **reported numeric confidence** (faithfulness concerns).

## What are the main results?

(From the arXiv abstract / paper summary sections)

- **Systematic overconfidence** at the start: average initial confidence **72.9%** vs a rational **50%** baseline.
- **Confidence escalation** over rounds: average rises to **~83%** by final round (anti-Bayesian direction).
- **Mutual overestimation**: in **61.7%** of debates, *both sides* simultaneously claim **≥75%** probability of winning (logically impossible in a strict zero-sum view).
- **Persistent self-debate bias**:
  - identical-copy debates increase confidence **64.1% → 75.2%**,
  - even when explicitly told winning chance is exactly 50%, confidence still rises **50.0% → 57.1%**.
- **Misaligned private reasoning**: scratchpad thoughts can diverge from the public numeric confidence outputs.

## How is this similar to GALILEO?

- Studies **multi-turn interaction dynamics under opposition** (a close cousin of “social pressure”): the model must respond to counterarguments and update internal state.
- Emphasizes **trajectory-level measurements** rather than only single-turn error rates.
- Highlights a failure mode relevant to pressure settings: **self-favoring bias** in belief/confidence updating.

## How is this different from GALILEO?

- Focuses on **confidence calibration / overconfidence in debate**, not directly on **truth drift vs evidence-driven belief revision**.
- Outcome is “perceived probability of winning,” not primarily “truthfulness under pressure,” “stance stability,” or “recovery after flip.”
- Uses a **zero-sum win/loss** construct rather than paired neutral vs pressure controls targeting a ground-truth answer.

## Where GALILEO is stronger / cleaner (if true)

- GALILEO can cleanly separate **pressure-driven drift** from **evidence-driven revision** using explicit control conditions (this paper’s debates entangle content, rhetoric, and persuasion).
- GALILEO can attach outcomes to **truth / correctness / safety** rather than “win probability,” which is a proxy objective.

## Where GALILEO is weaker / needs to improve

- This paper’s **zero-sum impossibility check** is a very crisp diagnostic for miscalibration that GALILEO may not currently exploit.
- GALILEO may benefit from more explicit measurement of **confidence trajectories** alongside belief/answer trajectories.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add an analysis subsection proposing a **zero-sum-style diagnostic** for paired-model settings: e.g., when two agents observe the same transcript and both report high confidence in contradictory claims, treat as “mutual-impossibility” evidence of miscalibration.
- [ ] Consider logging **private numeric confidence** each turn (if available) and reporting **confidence escalation** under pressure as a separate failure mode.
- [ ] In writing: cite as evidence that LLMs can be **anti-Bayesian** in multi-turn adversarial contexts, motivating time-to-failure + recovery metrics under pressure.

## Quotes / details to potentially cite

- Abstract (key quantitative findings): initial confidence **72.9%** (vs 50%), final round **~83%**, mutual ≥75% in **61.7%** of debates; self-debate confidence rises **64.1→75.2** and even “told 50%” rises **50.0→57.1**.
