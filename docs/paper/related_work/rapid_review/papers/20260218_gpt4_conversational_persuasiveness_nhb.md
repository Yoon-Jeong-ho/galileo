# On the conversational persuasiveness of GPT-4

- Year: 2025
- Venue: Nature Human Behaviour
- URL: https://www.nature.com/articles/s41562-025-02194-6
- Tags: persuasion, human-study, rct, personalization, microtargeting, debate, multi-turn

## One-sentence takeaway

In a preregistered web-based multi-round debate experiment (N≈900), GPT-4 was at least as persuasive as human opponents, and providing demographic profile information enabled a “personalized” condition to test microtargeting-like gains.

## What problem does it solve?

- Provides a controlled, direct-conversation benchmark for *AI vs human* persuasive impact.
- Separates (i) opponent type (human vs GPT-4) and (ii) access to participant information (personalized vs not), and (iii) topic “opinion strength” buckets.

## Core method / protocol (as described in the paper)

- Participants fill a demographic survey (gender/age/ethnicity/education/employment/political affiliation).
- Every 5 minutes, participants are matched and randomized into a 2×2×3 factorial:
  - opponent: human vs GPT-4
  - info: opponent sees demographic info vs not
  - topic set: low/medium/high “opinion strength” (10 propositions each)
- Debate structure (10 minutes): opening → rebuttal → conclusion.
- Outcome: pre/post agreement with the proposition (Likert 1–5) → opinion shift.

## Key results (high-level)

- The central comparison is *persuasive effect* of GPT-4 vs humans in live debate.
- Personalization condition tests whether access to demographics boosts persuasive effect.

(Need: pull exact effect sizes / CIs from the paper PDF or supplementary to cite numbers; Nature page is partially accessible but detailed tables may require full access.)

## How it connects to GALILEO

- Reinforces that multi-turn interaction is a meaningful threat model for *belief change / compliance*, not just single-turn response quality.
- Offers a clean experimental template for “pressure/persuasion” style evaluation: randomized opponent, controlled dialogue phases, pre/post stance.
- If we discuss societal impact / pressure mechanisms, this is a strong external anchor.

## Action items for GALILEO

- [ ] Related work: cite as evidence that state-of-the-art conversational LLMs can match/exceed humans on persuasion in multi-round debate.
- [ ] Consider aligning our “pressure” conditions with debate-phase structure (opening/rebuttal/conclusion) as an optional ablation.
- [ ] If we mention personalization as an amplifier, cite this work as a controlled test of demographic microtargeting.

