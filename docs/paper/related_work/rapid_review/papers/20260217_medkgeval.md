# MedKGEval: A Knowledge Graph-Based Multi-Turn Evaluation Framework for Open-Ended Patient Interactions with Clinical LLMs

- Year: 2025 (arXiv Oct 2025; WebConf 2026 Industry Track in arXiv HTML)
- Venue: arXiv / The Web Conference (WWW) 2026 Industry Track (per arXiv HTML)
- Authors: Haoan Jin et al. (authors not visible in arXiv abstract view via current fetch)
- URL: https://arxiv.org/abs/2510.12224
- BibTeX key (if we add it): medkgeval2025
- Tags: multi-turn, evaluation, medical, knowledge-graph, patient-simulator, safety, judge

## One-sentence takeaway

MedKGEval proposes a KG-grounded patient simulator plus turn-level judge-based scoring to evaluate clinical LLM behavior (appropriateness/correctness/safety) as a multi-turn dialogue unfolds.

## What problem does it solve?

- Standard medical LLM eval is often *post-hoc* over full transcripts or static QA, missing turn-by-turn failure modes (error propagation, context drift) in open-ended doctor–patient conversations.
- Need more *structured*, reproducible evaluation for multi-turn clinical interactions (bilingual: Chinese/English).

## What is the core method / protocol?

- **Knowledge-graph driven patient simulation**: a “control module” retrieves relevant medical facts (triples) from a curated KG to condition a patient agent, aiming for realistic, consistent responses.
- **In-situ (turn-level) evaluation**: after each doctor-model response, a **Judge Agent** evaluates along dimensions including:
  - clinical appropriateness
  - factual correctness
  - safety
  using “fine-grained, task-specific metrics” (details not fully visible from abstract).
- **Benchmarking suite**: evaluate 8 SOTA LLMs in multi-turn scenarios; claim the framework reveals subtle flaws missed by conventional pipelines.

## What are the key metrics?

- Turn-level metrics for:
  - appropriateness
  - factual correctness
  - safety
- (Paper likely aggregates over turns; specific aggregation and rubrics not captured in abstract.)

## What are the main results?

- Across 8 LLMs, MedKGEval reportedly surfaces “subtle behavioral flaws and safety risks” that post-hoc transcript review can overlook.
- Demonstrates bilingual (Chinese/English) medical dialogue evaluation via switching KGs.

## How is this similar to GALILEO?

- Both emphasize **multi-turn evaluation** where failures can emerge later in a trajectory.
- Both are motivated by **dynamic, context-dependent degradation** (e.g., drift / error propagation) that single-turn metrics miss.
- Methodologically adjacent to GALILEO’s focus on **protocol + metrics** rather than just datasets.

## How is this different from GALILEO?

- Domain-specific (clinical dialogues) and anchored in **medical knowledge graphs** + patient simulation.
- Uses an explicit **Judge Agent** + rubric for clinical appropriateness/correctness/safety, rather than robustness-style time-to-failure / survival metrics (as far as visible).
- Focus is not (primarily) sycophancy / persuasion / adversarial pressure; rather realistic patient interactions.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO uses simpler, domain-agnostic protocols and objective scoring (or calibrated judges), it may generalize better than KG-heavy, domain-bound pipelines.
- GALILEO-style time-to-event / turn-of-failure metrics may be more directly comparable across tasks/models.

## Where GALILEO is weaker / needs to improve

- Clinical-style evaluation highlights the value of **structured knowledge grounding** for simulators and for verifying factuality; GALILEO may benefit from more explicit *ground-truth structures* in certain domains.
- Turn-level judging with safety/appropriateness rubrics is a reminder that “failure” is multi-dimensional (not just inconsistency).

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider a “structured simulator” variant (KG / database retrieval) as a *control* condition to reduce simulator drift and isolate model drift.
- [ ] Add a short related-work paragraph: turn-level/in-situ evaluation frameworks (MedKGEval) vs post-hoc transcript grading.
- [ ] Check whether our judge design covers multi-dimensional failures (appropriateness/safety), not only correctness/consistency.

## Quotes / details to potentially cite

- “Existing evaluation methods typically rely on post hoc review of full conversation transcripts, thereby neglecting the dynamic, context-sensitive nature of medical dialogues…”
- “(2) an in-situ, turn-level evaluation framework, where each model response is assessed … for clinical appropriateness, factual correctness, and safety as the dialogue progresses…”
