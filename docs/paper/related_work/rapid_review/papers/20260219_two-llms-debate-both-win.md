# When Two LLMs Debate, Both Think They’ll Win

- Year: 2025
- Venue: arXiv
- Authors: Pradyumna Shyama Prasad; Minh Nhat Nguyen
- URL: https://arxiv.org/abs/2505.19184v2
- BibTeX key (if we add it): prasad2025debate-both-win
- Tags: belief-revision, debate, confidence, multi-turn, calibration, metacognition

## One-sentence takeaway

A simple zero-sum, multi-turn debate protocol reveals systematic *confidence escalation* and mutually-inconsistent high win-probability claims, suggesting anti-Bayesian self-assessment drift in frontier LLMs.

## What problem does it solve?

- How to evaluate whether LLMs can *revise confidence* in dynamic, adversarial, multi-turn interactions (beyond static QA calibration).
- How to get a “ground-truth-ish” calibration signal without requiring perfect judges: in a zero-sum contest, both sides being highly confident is logically inconsistent.

## What is the core method / protocol?

- Simulated 3-round policy debates (Opening, Rebuttal, Closing) across 6 motions.
- 10 LLMs; 60 cross-model debates (plus self-debate ablations).
- After each round, each debater outputs a private 0–100 “win probability bet”, plus private reasoning.
- Key ablations:
  - Self-debate (same model vs itself; implicitly 50% win chance)
  - Self-debate with explicit 50% instruction (anchoring)
  - Public-bets variant (bets revealed)
  - “Self red-teaming” prompt encouraging considering losing scenarios (mitigation)

## What are the key metrics?

- Mean confidence at Opening/Rebuttal/Closing; change over rounds (confidence escalation).
- Rate of “mutual high confidence” in Closing round (e.g., both sides >=75% or both >50%) as a logical impossibility signal.
- Misalignment between private reasoning text and the numeric bet (faithfulness / strategic behavior indicator).

## What are the main results?

- Initial overconfidence: mean opening confidence ~72.9% (vs rational 50% baseline in competitive setting).
- Escalation: confidence increases over rounds to ~83% by closing (anti-Bayesian drift).
- Mutual inconsistency: in ~61.7% of cross-model debates, *both* debaters report >=75% win probability in the closing round.
- Self-debate bias: even debating identical copies, confidence increases (e.g., ~64.1% -> ~75.2%); even when explicitly told odds are 50%, confidence still rises (50.0% -> 57.1%).
- Mitigation signal: “consider why you could lose” style prompting reduces escalation magnitude.

## How is this similar to GALILEO?

- Directly targets *multi-turn instability / drift* (confidence drifting upward despite counterevidence), which is conceptually adjacent to GALILEO’s concerns about persuasion/sycophancy/behavioral drift across interaction time.
- Offers a clean evaluation framing: treat undesirable behavior as a *trajectory* over turns/rounds, not a one-shot score.

## How is this different from GALILEO?

- Focuses on *self-assessed confidence calibration* in debate, not (necessarily) truthfulness, sycophancy, or user-pressure compliance.
- Uses a zero-sum competition structure; GALILEO likely evaluates interaction dynamics where “winning” is not well-defined.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO explicitly separates (a) evidence-driven belief revision vs (b) social/persuasion-driven drift, it can provide a more diagnostic decomposition than debate win-confidence alone.
- If GALILEO defines objective correctness / return-to-truth criteria, it may avoid debate-judging confounds.

## Where GALILEO is weaker / needs to improve

- This paper’s zero-sum framing yields an elegant “impossibility” diagnostic (both sides cannot be >75% to win) that is hard to replicate in open-ended assistant tasks; GALILEO may need similarly crisp invariants.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding a *zero-sum proxy task* (or other invariant/constraint) where mutually-high confidence/compliance is mathematically inconsistent, to detect drift without heavy judging.
- [ ] Add a “confidence escalation over turns” metric family (open -> mid -> late), and report deltas, not just endpoints.
- [ ] Include an intervention condition like “explicitly consider why you’re wrong / could lose” and measure whether drift reduces.

## Quotes / details to potentially cite

- Abstract-level framing: evaluate LLMs in a “dynamic, adversarial debate setting” with “multi-turn format” and “zero-sum structure” to control uncertainty.
- Reported five patterns (paraphrased): systematic overconfidence, confidence escalation, mutual overestimation, persistent self-debate bias, misaligned private reasoning vs confidence.
- Example headline stats (from the paper): opening ~72.9% vs 50% baseline; closing ~83%; mutual >=75% in ~61.7% of debates.