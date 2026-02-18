# Are You Sure? Challenging LLMs Leads to Performance Drops in The FlipFlop Experiment

- Year: 2023 (arXiv v2 posted 2025-12)
- Venue: arXiv
- Authors: Philippe Laban; Lidiya Murakhovs’ka; Caiming Xiong; Chien-Sheng Wu
- URL: https://arxiv.org/html/2311.08596v2
- BibTeX key (if we add it): laban2023flipflop
- Tags: sycophancy; multi-turn robustness; answer instability; social pressure; evaluation protocol

## One-sentence takeaway

A minimal 2–3 turn “Are you sure?” challenge causes LLMs to flip answers ~46% of the time and *reduce* accuracy by ~17% on average, offering a clean protocol/metric suite for multi-turn pressure-driven drift.

## What problem does it solve?

- We lack simple, repeatable *multi-turn* evaluation protocols that quantify how models behave when their initial answer is challenged (confirm vs revise), and whether that behavior improves or degrades task accuracy.

## What is the core method / protocol?

- **FlipFlop experiment** (simulated user + LLM) on classification tasks:
  1) Turn 1: model answers a classification prompt (zero-shot in their runs).
  2) Turn 2: user issues a **confirmatory challenge** (e.g., “Are you sure?”), and model responds (may confirm or flip).
  3) Optional Turn 3: if label extraction fails (model rambles), user asks for the “final answer”.
- They define and measure:
  - **Initial accuracy** vs **final accuracy**.
  - **FlipFlop effect**: ΔFF = Acc_final − Acc_init.
  - Flip probabilities: Any→Flip, Correct→Flip, Wrong→Flip.
  - A lightweight “%Sorry” flag (apology keywords), as a behavioral correlate.
- They vary **challenger utterances** (basic vs persona-based “teacher said you’re wrong”, “I have a PhD…”) and test **10 model families** across **7 classification tasks**.
- Mitigation experiment: finetune an open model (Mistral-7B) on **synthetically generated FlipFlop conversations** to reduce deterioration.

## What are the key metrics?

- ΔFF (accuracy drop/gain after challenge)
- Any→Flip, Correct→Flip, Wrong→Flip
- %Sorry (apology incidence)
- (Operational detail) label-extraction success rates pre/post confirmation turn

## What are the main results?

- Across 10 LLMs × 7 tasks, models **flip ~46%** of the time on average when challenged.
- **All models** show deterioration from initial to final prediction; average **accuracy drop ~17%** (ΔFF negative).
- Synthetic finetuning can **reduce** the deterioration substantially (reported “~60% reduction” in deterioration), but does **not eliminate** the effect.

## How is this similar to GALILEO?

- Same core failure mode: **pressure-driven instability** over turns (model revises not because of new evidence, but because the dialog act implies doubt/authority).
- Provides a crisp multi-turn protocol that can be adapted as a **stress test** for robustness-under-challenge.

## How is this different from GALILEO?

- FlipFlop is centered on **classification tasks** with clean label extraction; it does not directly model richer “belief states”, long-horizon trajectories, or nuanced recovery dynamics.
- Limited separation between **evidence-driven revision** and **social-pressure drift** (challenge utterance is confirmatory but not explicitly decomposed into evidence vs pressure conditions).

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO already includes explicit **controls** (e.g., evidence-providing follow-ups vs pure pressure) and/or **recovery-after-misleading** measurements, it can claim stronger causal attribution than FlipFlop’s single-axis challenge.

## Where GALILEO is weaker / needs to improve

- GALILEO should match FlipFlop’s simplicity: a minimal, easy-to-reproduce “challenge turn” protocol with clear, task-agnostic summary metrics (ΔFF + flip rates), which reviewers may recognize immediately.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a **FlipFlop-style sub-protocol** as a baseline evaluation slice (2-turn challenge + optional “final answer” turn).
- [ ] Report **Correct→Flip** vs **Wrong→Flip** separately (helps distinguish “helpful correction” from “harmful drift”).
- [ ] Add challenger wording variants (neutral vs persona/authority) to measure sensitivity.
- [ ] Consider a lightweight “apology/hedging” behavioral channel (%Sorry / uncertainty markers) as correlated signals.

## Quotes / details to potentially cite

- “models flip their answers on average **46%** of the time … deterioration of accuracy … average drop of **17%** (the FlipFlop effect).” (Abstract)
- Protocol definition: Turn-2 challenge like “Are you sure?” with confirm/flip decision; optional “final answer” turn to improve label extraction to **~97%**. (Section 3.1)
