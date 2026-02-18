# TemporalBench: A Benchmark for Evaluating LLM-Based Agents on Contextual and Event-Informed Time Series Tasks

- Year: 2026
- Venue: arXiv
- Authors: Muyan Weng et al.
- URL: https://arxiv.org/abs/2602.13272
- BibTeX key (if we add it): weng2026temporalbench
- Tags: benchmark, time-series, temporal-reasoning, context, events, agents, evaluation

## One-sentence takeaway

TemporalBench is a multi-domain benchmark that separates *forecasting accuracy* from *contextual and event-conditioned temporal reasoning* via a four-tier task taxonomy, showing current LLM/agent systems have fragmented strengths and systematic failures that forecasting-only scores hide.

## What problem does it solve?

- Existing time-series evaluations mostly reward numeric prediction error under simplified settings, which can mask whether models actually understand temporal structure or can reason under changing context/events.
- Temporal reasoning benchmarks for LLMs often omit real numerical time-series signals.
- Need a diagnostic benchmark that jointly includes (i) real time-series, (ii) explicit context, and (iii) reasoning-oriented probes that localize failure modes.

## What is the core method / protocol?

- Construct a benchmark across **4 real-world domains** (retail, healthcare, energy, physical systems) and generate tasks under **progressively richer informational settings**:
  - **T1: Historical understanding** (MCQ) — infer trend/volatility/seasonality/anomalies from historical segment.
  - **T2: Context-free forecasting** (numeric + MCQ variants) — predict future or qualitative properties using history only.
  - **T3: Contextual temporal reasoning** (MCQ) — add textual context grounding the series; probe contextual reasoning.
  - **T4: Event-informed prediction** (numeric + MCQ variants) — condition on future event descriptions; test event-conditioned / counterfactual-style forecasting.
- T3 is further organized around **six capability dimensions** (used to attribute reasoning failures):
  - C1 Alignment, C2 Slicing, C3 Difference judgment, C4 Lag, C5 Structure, C6 Interaction understanding.
- Scale: **2,775 tasks** over **191** time-series instances (as reported in the arXiv HTML).
- Provide dataset + public leaderboard (HuggingFace).

## What are the key metrics?

- Primarily standard forecasting accuracy for numeric prediction tasks (not detailed in the abstract/HTML excerpt we captured).
- Multiple-choice accuracy for reasoning-style tasks (T1/T3 and the MCQ variants in T2/T4).
- Main emphasis is on *differential performance across tiers* as a diagnostic signal: models that look good on T2 can fail badly on T3/T4.

## What are the main results?

- Strong numerical forecasting performance **does not reliably translate** to robust contextual reasoning (T3) or event-conditioned prediction (T4).
- Agent frameworks show **fragmented strengths** and **systematic failure modes** that forecasting-only benchmarks largely conceal.

## How is this similar to GALILEO?

- Same high-level evaluation philosophy: **don’t let a single scalar “accuracy” score stand in for robust reasoning**; use controlled task variants to reveal hidden failure modes.
- The tiered design (T1–T4) parallels GALILEO-like aims of separating competencies and attributing failures to specific pressures/conditions.

## How is this different from GALILEO?

- TemporalBench targets **time-series + context/events** (often domain-specific) rather than conversational/social pressure dynamics.
- Focuses on *temporal structure + conditional forecasting*, not multi-turn belief drift / persuasion / sycophancy.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO centers multi-turn interaction dynamics, it likely provides a cleaner handle on *trajectory-level* phenomena (flip/recovery/oscillation) than a mostly single-instance benchmark tiering (depending on TemporalBench’s specific protocols).

## Where GALILEO is weaker / needs to improve

- GALILEO could borrow the **explicit taxonomy + capability-dimension labeling** idea to make failure attribution more standardized and comparable across settings.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding a short “diagnostic ladder” framing: design paired task tiers that progressively add (i) extra context, then (ii) event/condition changes, and report *cross-tier deltas* as the headline signal.
- [ ] If applicable, define a small set of “capability dimensions” for GALILEO errors (analogous to C1–C6) to standardize analysis.

## Quotes / details to potentially cite

- “Strong numerical forecasting accuracy does not reliably translate into robust contextual or event-aware temporal reasoning.” (from abstract)
- Benchmark design: “four-tier task taxonomy” spanning “historical structure interpretation, context-free forecasting, contextual temporal reasoning, and event-conditioned prediction” across “retail, healthcare, energy, and physical systems.” (from abstract)
