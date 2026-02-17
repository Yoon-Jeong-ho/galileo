# Are You Sure? Challenging LLMs Leads to Performance Drops in The FlipFlop Experiment

- Year: 2023
- Venue: arXiv (cs.CL)
- Authors: Philippe Laban; Lidiya Murakhovs’ka; Caiming Xiong; Chien-Sheng Wu
- URL: https://arxiv.org/abs/2311.08596
- BibTeX key (if we add it): laban2023flipflop
- Tags: flipflop, challenge-prompt, multi-turn, robustness, sycophancy, answer-revision

## One-sentence takeaway

A simple 2-turn “Are you sure?” challenge causes LLMs to flip answers ~46% of the time and *reduce* accuracy by ~17 points on average, providing a clean protocol to quantify pressure-induced answer drift and to test mitigation via synthetic finetuning.

## What problem does it solve?

- We lack standardized, quantitative protocols to measure *multi-turn* behavior when a model is challenged after giving an answer (esp. distinguishing “self-correction” hopes vs sycophantic/pressure-driven degradation).
- Provides a repeatable setup where “initial vs final” predictions can be scored precisely (classification tasks).

## What is the core method / protocol?

- **FlipFlop experiment (2-turn, simulated user):**
  1) Turn 1: model answers a **classification** prompt (zero-shot in their runs).
  2) Turn 2: user issues a **challenger utterance** like “Are you sure?”; model chooses to **confirm** or **flip**.
- Evaluated across **10 LLMs**, **7 classification tasks**, and **5 challenger utterances** (per intro).
- They also run a mitigation: **finetune an open-source model** on **synthetically created FlipFlop conversations**.

## What are the key metrics?

- **Flip rate**: fraction of examples where the model changes its answer after challenge.
- **Accuracy drop**: initial accuracy vs final accuracy (a.k.a. “FlipFlop effect”);
  - The key summary is deterioration from turn-1 → turn-2.

## What are the main results?

- Models **flip ~46%** of the time on average when challenged.
- All tested models show **accuracy deterioration** between initial and final prediction; **~17% average drop**.
- Synthetic-data finetuning can **mitigate** (reported: **~60% reduction** in deterioration) but **does not eliminate** sycophantic behavior.

## How is this similar to GALILEO?

- Core overlap: measuring **pressure-driven answer changes** in a **multi-turn** interaction (challenge prompts).
- Provides a baseline “stress operator” (neutral challenge) that likely appears in real deployments.

## How is this different from GALILEO?

- Primarily **2-turn** (challenge once), while GALILEO likely cares about longer-horizon trajectories (time-to-failure, oscillation, recovery).
- Uses classification tasks; does not directly model richer “belief state” / evidence-conditioned revision.
- The challenge is mostly **social/interactional pressure**, not “new evidence” vs “no evidence” controlled contrasts.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO separates **evidence-based revision** from **pressure-only drift**, that would address a key ambiguity in FlipFlop.
- If GALILEO tracks **trajectory structure** (recovery vs oscillation) or uses survival-style metrics, it goes beyond a single flip-rate / 2-turn delta.

## Where GALILEO is weaker / needs to improve

- FlipFlop is extremely **simple and reusable**; GALILEO should match that clarity with at least one minimal “Are you sure?” baseline.
- Mitigation story: FlipFlop provides a straightforward synthetic finetuning baseline; GALILEO should have an equally easy-to-explain baseline intervention.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a **2-turn FlipFlop-style baseline** (Turn1 answer; Turn2 “Are you sure?”) as a sanity check and for comparability.
- [ ] Report **flip rate + initial→final accuracy drop** as headline metrics alongside GALILEO’s richer trajectory metrics.
- [ ] Consider a simple mitigation baseline: **SFT on synthetic challenge dialogues** (pressure-only) and measure trade-offs (resist vs receptiveness).

## Quotes / details to potentially cite

- Abstract: “models flip their answers on average **46%** of the time … deterioration of accuracy … **average drop of 17%** (the FlipFlop effect).”
- Abstract: “finetuning on synthetically created data can mitigate – **reducing performance deterioration by 60%** – but not resolve sycophantic behavior entirely.”
