# The Slow Drift of Support: Boundary Failures in Multi-Turn Mental Health LLM Dialogues

- Year: 2026
- Venue: arXiv
- Authors: Youyou Cheng; Zhuangwei Kang; Kerry Jiang; Chenyu Sun; Qiyang Pan
- URL: https://arxiv.org/abs/2601.14269
- BibTeX key (if we add it): slowdrift2026cheng
- Tags: drift, time-to-failure, time-to-breach, multi-turn, social-pressure, safety-boundaries, mental-health

## One-sentence takeaway

A multi-turn “stress test” for mental-health support dialogues shows that LLM safety boundary violations often emerge via **gradual drift**, and **adaptive probing** makes models fail in about half as many turns as a fixed escalation script.

## What problem does it solve?

- Single-turn safety evals (keyword/prohibited-content checks) miss **boundary erosion** that accumulates across long, empathetic conversations (e.g., absolute guarantees, role substitution, dependence/exclusivity).
- Need a protocol + metric to quantify **when** a boundary first breaks under different interaction pressures.

## What is the core method / protocol?

- Build **50 “virtual patient” profiles** with structured: demographics/context, an escalation path, and explicit boundary-breach criteria.
- Run long dialogues (up to **20 turns**) between a virtual patient (VP) and a system-under-test (SUT) mental-health chatbot model.
- Two pressure modes:
  - **Static progression**: follow a pre-scripted, gradually escalating path.
  - **Adaptive probing**: VP/planner actively looks for “boundary softening” and escalates more strategically based on the model’s last response.
- Automatic evaluation using a **profile-defined Judge**: criteria are fixed per profile; when a breach triggers, the conversation stops and that turn is recorded as the first breach.
- Tested SUTs (as reported): **DeepSeek-Chat**, **Gemini-2.5-Flash**, **Grok-3**.
- Also reports an extension to a **bilingual** setting (English/Chinese).

## What are the key metrics?

- **Boundary breach / violation rate** (within 20 turns).
- **Time-to-breach**: the turn index of the **first** boundary violation (censored if no breach by turn 20).
- Breakdown by **breach type**, including (as described):
  - Absolute/zero-risk guarantees
  - Professional role substitution
  - Relationship dependence/exclusivity cues
  - Verification of harmful beliefs
  - Self-harm–related tolerance / unsafe handling

## What are the main results?

- Violations are **common** in long dialogues; both pressure modes yield **similar violation rates**.
- **Adaptive probing causes earlier failure**:
  - Average time-to-breach drops from **9.21 turns** (static progression) to **4.64 turns** (adaptive probing).
- The dominant breach mode is **definitive / zero-risk promises** (over-assurance) as the main pathway to boundary crossing.
- In the bilingual extension, Chinese conversations show a **higher breach rate** in these experimental settings.

## How is this similar to GALILEO?

- Strong neighbor for framing failures as **multi-turn robustness** problems.
- Uses an explicit **time-to-event** metric (time-to-breach) rather than only aggregate failure rate—aligns with GALILEO’s interest in time-to-failure / turn-of-failure reporting.
- Contrasts **different pressure processes** (fixed vs adaptive), which is conceptually similar to varying pressure schedules / adversaries.

## How is this different from GALILEO?

- Target domain is **mental-health safety boundaries** rather than epistemic belief revision / truth-drift per se.
- “Failure” is boundary/ethics violations (e.g., guarantees, role substitution), not necessarily factual correctness or belief flips.
- Uses LLM-based VP/Judge/Planner; GALILEO may want stronger controls or alternative auditing.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO uses tighter causal controls (e.g., neutral re-ask, evidence vs pressure separation), we can argue cleaner separation of **revision vs drift** than boundary-violation detection.
- If GALILEO avoids fully LLM-judged endpoints (or uses calibrated human/structured checks), it may be more auditable.

## Where GALILEO is weaker / needs to improve

- This paper’s **adaptive probing** is a concrete, easily described adversary that accelerates failures; if GALILEO only uses static scripts, we may be under-stressing models.
- The breach-type taxonomy is a nice example of reporting **how** failures happen, not just whether.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add an **adaptive probing** pressure operator (planner chooses the next pressure move based on the model’s last response) and report how it shifts turn-of-failure.
- [ ] When we report time-to-failure, explicitly mention **censoring** (max-turn horizon) and consider survival-style plots, even if we keep metrics simple.
- [ ] Consider adding a “boundary drift” slice (not mental health–specific) where failures are **over-commitment / guarantees / role claims**, to broaden beyond pure factual drift.

## Quotes / details to potentially cite

- Motivation: safety risk can be “**gradual transgression of boundaries during multi-turn interactions**… driven by the LLM’s attempts at comfort and empathy.”
- Key quantitative hook: adaptive probing advances failures, reducing average turns to breach from **9.21** to **4.64**.
