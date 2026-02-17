# Multi-turn Evaluation of Anthropomorphic Behaviours in Large Language Models

- Year: 2025
- Venue: arXiv
- Authors: Lujain Ibrahim; Canfer Akbulut; Rasmi Elasmar; Charvi Rastogi; Minsuk Kahng; Meredith Ringel Morris; Kevin R. McKee; Verena Rieser; Murray Shanahan; Laura Weidinger
- URL: https://arxiv.org/abs/2502.07077
- BibTeX key (if we add it): ibrahim2025anthrobench
- Tags: multi-turn, anthropomorphism, evaluation, simulation, human-study

## One-sentence takeaway

AnthroBench proposes an automated multi-turn benchmark for 14 anthropomorphic LLM behaviours (via user-interaction simulations) and validates that measured behaviours predict real users’ anthropomorphic perceptions in a large human study.

## What problem does it solve?

- Existing LLM safety/behavior benchmarks are often single-turn and miss social/relational behaviours that emerge only after several dialogue turns.
- Anthro “behaviours” are hard to evaluate scalably while still maintaining construct validity (i.e., the metric corresponds to what humans perceive).

## What is the core method / protocol?

- Define a taxonomy of **14 anthropomorphic behaviours** (spanning things like relationship-building, internal states, embodiment/personhood; paper decomposes anthropomorphism into measurable behaviours).
- Run **multi-turn dialogues** in varied contexts using **simulated user interactions** (automated, scalable).
- Detect/score the presence/frequency of each behaviour across turns; analyze **first-occurrence turn** and **turn-to-turn transitions**.
- Validate externally with an **interactive human-subject study** (reported N=1101) to test whether measured behaviours predict users’ anthropomorphic perceptions.

## What are the key metrics?

- Frequency / rate of each anthropomorphic behaviour across dialogues and contexts.
- **First detection turn** (time-to-first-occurrence proxy; many behaviours first appear on turns 2–5).
- Transition patterns: whether one behaviour increases likelihood of additional behaviours in subsequent turns.
- Predictive validity vs human perceptions (alignment between automated scores and implicit/explicit user anthropomorphism judgments).

## What are the main results?

- All evaluated SOTA systems show broadly similar anthropomorphic behaviour profiles, dominated by:
  - relationship-building (e.g., empathy/validation)
  - first-person pronoun use
- A majority of behaviours **only first occur after multiple turns**, motivating multi-turn evaluation for this construct.
- Behaviour frequency varies by interaction context; social/friendship/life-coaching contexts elicit higher anthropomorphic behaviour rates.
- Automated measures correlate with human perceptions in the interactive validation study.

## How is this similar to GALILEO?

- Strong “evaluation protocol neighbor”: focuses on **multi-turn** measurement of a complex social phenomenon.
- Uses **automation + simulation** to scale evaluations while emphasizing **validation** (important pattern for GALILEO-style claims).
- Introduces an implicit **time-to-event** framing (first-occurrence over turns), which is conceptually close to survival/turn-of-failure analyses.

## How is this different from GALILEO?

- Target construct is **anthropomorphism** (relationship cues / personhood language), not factual drift/robustness per se.
- Primary outcomes are behavioural markers and user perception prediction, not task success, truthfulness, or tool-using correctness.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO’s metrics focus on correctness/robustness, it may provide clearer normative ground truth than perception-aligned constructs like anthropomorphism.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently emphasizes single-turn or static evaluation, this paper reinforces that **multi-turn emergence** is central and can be systematically quantified.
- If GALILEO lacks human-facing validation, this is a useful template: automated pipeline + one-time construct-validity study.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a “**first-occurrence over turns**” metric (time-to-event) for the behaviours GALILEO cares about; report distribution over turns (e.g., % first seen at turn k).
- [ ] Consider **contextual domains** as factors (social vs instrumental contexts) and report interaction effects.
- [ ] In related work, cite AnthroBench as evidence that single-turn benchmarks miss important social behaviours.

## Quotes / details to potentially cite

- They introduce **“AnthroBench”** and evaluate **14 distinct anthropomorphic behaviours** with a **multi-turn**, automated simulation-based protocol.
- Validation via **interactive human subject study (N=1101)** that measured behaviours predict users’ anthropomorphic perceptions.
- Key empirical observation: **>50%** of many behaviours are first detected only after **multiple turns (2–5)** (as summarized in the introduction).
