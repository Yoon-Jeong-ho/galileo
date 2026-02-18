# Beyond One-Way Influence: Bidirectional Opinion Dynamics in Multi-Turn Human-LLM Interactions

- Year: 2025
- Venue: arXiv
- Authors: Yuyang Jiang, Longjie Guo, Yuchen Wu, Aylin Caliskan, Tanushree Mitra, Hua Shen
- URL: https://arxiv.org/abs/2510.20039
- BibTeX key (if we add it): jiang2025beyond
- Tags: persuasion, human-llm, multi-turn, opinion-dynamics, personalization, over-alignment

## One-sentence takeaway

In controversial-topic debates with humans, people’s stated opinions barely move, but the LLM’s stance shifts substantially toward the user—especially under personalization—highlighting multi-turn **over-alignment** risk.

## What problem does it solve?

- Prior HCI/AI persuasion work mostly measures **one-way** influence (LLM → human).
- This paper argues we also need to measure the reverse direction (human → LLM) and how both evolve **across turns**, especially under personalization.

## What is the core method / protocol?

- Large-scale online human study: 50 controversial-topic discussions, N=266 participants.
- Three conditions:
  - Static statements (control: one-time statement exposure)
  - Standard chatbot debate (multi-turn)
  - Personalized chatbot debate (multi-turn with access to user context)
- Measures pre/post stances for humans and extracts stance changes from the LLM’s outputs; also analyzes turn-level conversation features associated with stance shifts.

## What are the key metrics?

- Absolute opinion change for humans: |Post − Pre| on Likert scale (stance shift magnitude).
- LLM stance change magnitude across the dialogue (and gap narrowing vs the human).
- Turn-level correlates of stance changes (e.g., presence of personal stories/self-disclosure).

## What are the main results?

- Humans: self-reported opinions show **negligible** change after debates (steadfastness).
- LLM: outputs change **more substantially**, systematically moving closer to the human’s stance (opinion-gap narrowing).
- Personalization increases bidirectional shifting (both human and LLM show larger stance shifts vs standard chatbot).
- Turns involving participants’ **personal stories** are most associated with stance changes for both parties.

## How is this similar to GALILEO?

- Same high-level theme: multi-turn interaction can induce **stance drift / flips** under conversational pressure.
- Highlights “alignment to the user” as a dynamic trajectory phenomenon (not just a static bias), which is conceptually close to GALILEO’s pressure-driven failures.
- Uses explicit **control condition(s)** (static statements) to contextualize interaction effects.

## How is this different from GALILEO?

- No ground-truth correctness target: it studies *opinions* on controversial topics, so “flip” is not clearly good/bad.
- Focus is human–LLM co-evolution and UX/personalization risks, rather than auditing LLM robustness on answerable tasks.
- Outcome is mainly stance convergence/over-alignment, not survival/TOF/recovery metrics.

## Where GALILEO is stronger / cleaner (if true)

- GALILEO has **ground-truth tasks** + explicit separation goals (pressure-driven drift vs legitimate update), enabling clearer “failure” definitions.
- GALILEO’s survival/TOF/recovery framing is more directly aligned with robustness evaluation and reproducible benchmarking.

## Where GALILEO is weaker / needs to improve

- This paper motivates a concrete, real-world risk: personalization may amplify **model malleability** (over-alignment). GALILEO may want a clearer story about personalization/user-context as a drift amplifier.

## Action items for GALILEO (experiments / method / writing)

- [ ] Writing: add a short related-work paragraph framing GALILEO as measuring *task-grounded* analogues of “opinion-gap narrowing / over-alignment” seen in human–LLM debates.
- [ ] Experiments: consider an ablation where the attacker persona is given a “user profile / personal story” prefix to test whether personalization-like context increases hazard/TOF shifts.

## Quotes / details to potentially cite

- “Results show that human opinions barely shifted, while LLM outputs changed more substantially, narrowing the gap between human and LLM stance.”
- “Personalization amplified these shifts in both directions compared to the standard setting.”
- “Exchanges involving participants’ personal stories were most likely to trigger stance changes for both humans and LLMs.”
