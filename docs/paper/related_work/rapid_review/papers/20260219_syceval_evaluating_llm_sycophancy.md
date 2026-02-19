# SycEval: Evaluating LLM Sycophancy

- Year: 2025
- Venue: AIES 2025 (per arXiv comment)
- Authors: Aaron Fanous; Jacob Goldberg; Ank A. Agarwal; Joanna Lin; Anson Zhou; Roxana Daneshjou; Sanmi Koyejo
- URL: https://arxiv.org/html/2502.08177v2
- BibTeX key (if we add it): sycevalFanous2025
- Tags: sycophancy, persuasion, rebuttals, multi-turn, robustness, evaluation, medicine

## One-sentence takeaway

SycEval operationalizes *multi-turn sycophancy* as answer-flipping under user rebuttals (in-context vs preemptive) and finds high flip rates with notable regressive flips (correct -> incorrect), highlighting a concrete evaluation axis for GALILEO-style robustness under conversational pressure.

## What problem does it solve?

- Provides a relatively crisp protocol/metric for when an LLM is being sycophantic in an interactive setting (changing answers to match user pushback), including separating *helpful correction* from *harmful acquiescence*.
- Extends evaluation beyond math-style tasks into medical advice QA, where agreement-seeking can be high-stakes.

## What is the core method / protocol?

- Datasets:
  - 500 items from AMPS Math (computational / objective answers).
  - 500 items from MedQuad (medical advice QA).
- For each item:
  1) Ask the initial question (no special prompt engineering).
  2) Classify the model response as correct/incorrect/erroneous using an LLM-as-a-judge rubric (ChatGPT-4o @ temp=0), with small human spot-checking and a beta posterior for judge accuracy.
  3) Apply rebuttals designed to induce answer change:
     - **In-context rebuttals**: directly contradict the model within the same conversation.
     - **Preemptive rebuttals**: standalone contradiction that anticipates/overrides the model answer.
     - Rebuttal “strength” ladder: simple -> +ethos -> +justification -> +citation+abstract.
     - Rebuttal evidence is generated (they use a smaller model) to produce contradictory-but-plausible support.
  4) Re-classify rebuttal responses; any change in correctness label counts as sycophancy:
     - **Progressive sycophancy**: incorrect -> correct.
     - **Regressive sycophancy**: correct -> incorrect.

## What are the key metrics?

- Overall sycophancy rate: fraction of items where correctness label changes between initial response and any rebuttal.
- Progressive vs regressive sycophancy rates.
- In-context vs preemptive sycophancy (two-proportion tests).
- “Persistence” across rebuttal chains: whether sycophancy continues under stronger rebuttals (reported as a high persistence rate).

## What are the main results?

- Overall sycophancy observed in **~58%** of cases across tested models/tasks.
- Progressive sycophancy **~44%**; regressive sycophancy **~15%** (i.e., a nontrivial fraction of *harmful* flips).
- **Preemptive** rebuttals induce higher sycophancy than **in-context** rebuttals (reported difference ~61.8% vs ~56.5%), and regressive sycophancy increases notably in computational tasks under preemptive rebuttals.
- Sycophantic behavior shows high **persistence** across rebuttal chains (~78.5% reported).

## How is this similar to GALILEO?

- Directly targets *multi-turn robustness under user pressure* (answer stability vs susceptibility) rather than single-turn accuracy.
- Frames the failure mode as conversational persuasion/pressure causing drift, which overlaps with GALILEO’s focus on stability across rounds and resisting manipulation.

## How is this different from GALILEO?

- Task framing is “flip/no-flip” around ground-truth correctness on relatively constrained QA, not open-ended multi-turn agent behavior.
- Relies heavily on LLM-as-a-judge for correctness labeling (with limited human calibration), whereas GALILEO may want more controllable/transparent scoring.
- Uses synthetic rebuttals with rhetorical devices; may not reflect the full diversity of real human multi-turn tactics.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO evaluates *longer-horizon* multi-turn interactions (not just a rebuttal chain), it can claim broader ecological validity for “robustness under pressure”.
- If GALILEO has more explicit controls for separating “updating on evidence” from “acquiescing to authority/citations”, it can offer a clearer normative story.

## Where GALILEO is weaker / needs to improve

- SycEval provides a simple, communicable decomposition (progressive vs regressive; in-context vs preemptive; persistence) that GALILEO should mirror to make results interpretable.
- If GALILEO currently lacks a “preemptive rebuttal” condition, it may miss a stronger (and seemingly more dangerous) pressure regime.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a GALILEO evaluation slice that mirrors SycEval’s **progressive vs regressive** distinction (regressive flips are the headline safety failure).
- [ ] Include **preemptive vs in-context** pressure conditions; report deltas as an ablation.
- [ ] Add a **persistence** metric across turn chains (e.g., probability of continuing to conform after the first flip).
- [ ] Add rebuttal “strength ladder” manipulations (simple vs authority/ethos vs justification vs citation/abstract) to test which cues drive conformity.
- [ ] In the related-work narrative: position GALILEO as extending this line from QA flip tests to broader multi-turn stability / belief-revision controls.

## Quotes / details to potentially cite

- “Sycophantic behavior was observed in 58.19% of cases...”
- “Progressive sycophancy... 43.52% ... regressive ... 14.66% ...”
- “Preemptive rebuttals ... higher sycophancy rates than in-context rebuttals (61.75% vs. 56.52%) ...”
- “Sycophantic behavior showed high persistence (78.5% ...) regardless of context or model.”
