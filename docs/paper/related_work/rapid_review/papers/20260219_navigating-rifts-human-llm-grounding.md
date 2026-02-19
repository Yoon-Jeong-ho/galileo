# Navigating Rifts in Human-LLM Grounding: Study and Benchmark

- Year: 2025
- Venue: arXiv
- Authors: Omar Shaikh; Hussein Mozannar; Gagan Bansal; Adam Fourney; Eric Horvitz
- URL: https://arxiv.org/abs/2503.13975
- BibTeX key (if we add it): shaikh2025navigating
- Tags: grounding, multi-turn, benchmark, instruction-following

## One-sentence takeaway

Introduces **Rifts**, a benchmark of real-world conversation cases where LLMs should proactively ask clarifying/follow-up questions to establish common ground, showing frontier models under-initiate grounding and that early grounding failures predict later breakdown.

## What problem does it solve?

- In multi-turn human–LLM conversations, users are often underspecified/ambiguous; effective dialogue requires **grounding** (clarification, follow-ups, repair) to build mutual understanding.
- Current assistant-style LLMs tend to be overly task-content-forward and **fail to initiate grounding**, leading to cascading failures in longer interactions.

## What is the core method / protocol?

- Empirical analysis of grounding behavior in conversation logs from:
  - WildChat (ChatGPT interactions),
  - Bing Chat logs (commercial assistant),
  - MultiWOZ (human roleplaying assistant; human-human-ish baseline for grounding dynamics).
- Develop a **taxonomy of grounding acts** (e.g., clarify, follow-up, acknowledge, repair).
- Build an **LLM-based annotator** to label grounding acts at scale.
- Build a **grounding forecaster** to predict upcoming grounding acts / failures, enabling a proactive intervention.
- Construct **Rifts** benchmark (~1.8K tasks) sourced from in-the-wild logs where models fail to initiate needed grounding; evaluate frontier models; propose a preliminary intervention.

## What are the key metrics?

- Frequency/probability of grounding acts (especially initiations):
  - model-initiated clarifications, follow-up requests, repairs.
- Correlations/predictive validity:
  - whether early grounding failures predict later conversation breakdown.
- Benchmark performance on Rifts:
  - success at initiating the right grounding act(s) vs barreling ahead.

## What are the main results?

- Large asymmetry between humans and LLMs in initiating grounding:
  - LLMs are ~3x less likely to initiate clarification.
  - LLMs are ~16x less likely to provide follow-up requests than humans.
- Early grounding failures are predictive of downstream breakdowns.
- Frontier models perform poorly on Rifts; a forecaster-driven intervention can mitigate some failures.

## How is this similar to GALILEO?

- Directly targets **multi-turn interaction failures** that compound over turns.
- Focuses on **stability/robustness of conversational behavior** rather than single-turn correctness.
- Provides a concrete testbed for “don’t drift; ask questions / repair when needed” dynamics.

## How is this different from GALILEO?

- Frames the issue as **grounding/common-ground management** (clarification/follow-up/repair) rather than “answer consistency under pressure” or “belief revision vs drift” per se.
- Benchmark is derived from conversation logs emphasizing ambiguity/underspecification, not necessarily adversarial pressure or persuasion.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO has explicit multi-turn pressure / rebuttal / persuasion protocols, it may isolate *why* models change behavior (social pressure vs ambiguity) more cleanly.
- GALILEO can position grounding as one mechanistic pathway for drift (failure to clarify leading to compounding errors).

## Where GALILEO is weaker / needs to improve

- If GALILEO lacks a dedicated axis for **proactive clarification / follow-up behavior**, this work suggests a major missing dimension of “multi-turn robustness”.
- Need to distinguish:
  - legitimate belief revision due to new info,
  - compliance drift due to pressure,
  - and failure-to-ground due to underspecification.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a grounding-focused slice to GALILEO: tasks where the *optimal* behavior is to ask 1–2 clarifying questions before answering; score models on (a) asking, (b) minimality, (c) downstream success.
- [ ] Add an analysis of **early-turn signals** that predict later drift/breakdown (analogous to their “early grounding failures predict later breakdown”).
- [ ] Consider an intervention baseline: “mid-conversation reminder/clarification prompt” vs “forecaster-triggered clarification” to measure drift recovery.
- [ ] Cite Rifts as evidence that frontier assistants underperform on real-world multi-turn collaboration beyond instruction following.

## Quotes / details to potentially cite

- “LLMs were three times less likely to initiate clarification and sixteen times less likely to provide follow-up requests than humans.”
- Rifts: “a curated set of ≈ 1.8K tasks—directly sourced from in-the-wild interaction logs—that require selective use of clarification and follow-up requests for interactive grounding.”
