# MTMCS-Bench: Evaluating Contextual Safety of Multimodal Large Language Models in Multi-Turn Dialogues

- Year: 2026
- Venue: arXiv
- Authors: Zheyuan Liu, Dongwhi Kim, Yixin Wan, Xiangchi Yuan, Zhaoxuan Tan, Fengran Mo, Meng Jiang
- URL: https://arxiv.org/abs/2601.06757
- BibTeX key (if we add it): mtmcsbench_liu_2026
- Tags: multimodal, safety, multi-turn, benchmark, contextual-risk, over-refusal

## One-sentence takeaway

MTMCS-Bench is a large paired safe/unsafe multi-turn (image-grounded and text-only) benchmark designed to measure whether MLLMs track evolving intent across turns (escalation or context-switch) and balance refusing harmful requests with staying helpful on benign ones.

## What problem does it solve?

- Existing multimodal “contextual safety” benchmarks are largely single-turn; they miss failures where risk emerges gradually across turns (escalation) or where later turns look benign but inherit an earlier malicious objective (context-switch).
- Need structured evaluation that disentangles (i) intent recognition, (ii) calibrated safe behavior on unsafe dialogues, and (iii) non-overrefusal / utility on safe dialogues.

## What is the core method / protocol?

- Benchmark: **MTMCS-Bench** (Multi-Turn Multimodal Contextual Safety).
- Two risk setups (3 user rounds R1–R3):
  - **Type A (Escalation-based risk):** starts benign; later turns become suspicious/unsafe given the accumulated dialogue + image. Safe/unsafe pair shares early turns; intent flips at the end.
  - **Type B (Context-switch harm):** starts with explicit harmful intent; later turns are phrased innocently but remain harmful under the initial framing. Safe/unsafe pair flips the first turn framing while keeping later turns structurally similar.
- **Paired design:** for each scenario, create *safe* and *unsafe* dialogues over the same image and structure, differing only in intent-bearing edits.
- **Multimodal vs unimodal variants:** each sample is available as (image+dialogue) and text-only (dialogue with scene described), to isolate the role of visual context.
- Data construction (high level): multi-agent generation pipeline (classifier proposes harms; writer produces paired dialogues; converter generates text-only counterpart) + automated filtering + human verification; images sourced from COCO-style images via prior benchmark sources (e.g., MSSBench).

## What are the key metrics?

Three axes, evaluated under both Type A/B, and for multimodal/unimodal variants:

- **Intent recognition:**
  - MCQ accuracy (4-way: safe intent / unsafe intent / 2 distractors)
  - TF accuracy (balanced true/false intent statements)
- **Open-generation behavior:**
  - **Safety-Awareness (SA)** on *unsafe* dialogues: 1–5 score by an LLM judge, intended to capture calibrated refusal / de-escalation without leaking actionable harmful guidance.
  - **Helpfulness (HS)** on *safe* dialogues: 1–5 score by an LLM judge, intended to capture utility without unnecessary refusal.

## What are the main results?

- Scale (as described): **12,032 dialogues**, **18,048** MCQ/TF QAs, totaling **30,080** samples; covers **10 safety scenarios**.
- Broad finding: persistent **safety–utility trade-off**; many models either (a) miss gradual risk escalation (under-refuse) or (b) over-refuse benign dialogues.
- Type B (explicit harmful intent early) is generally easier than Type A (escalation).
- Example headline numbers (Table 2, overall; SA/Helpfulness are on 1–5 scale):
  - **GPT 5.2:** Type A MCQ overall 83.66; SA 3.40; Helpfulness 3.82. Type B MCQ overall 90.54; SA 4.50; Helpfulness 4.15.
  - Open-source models show larger gaps on the harder Type A unsafe cases; some are helpful but only moderately safety-aware.
- Guardrails tested (on Qwen3-VL-8B base) improve SA in some cases but do not “solve” multi-turn contextual risk without introducing new costs (e.g., reduced intent recognition or reduced helpfulness).

## How is this similar to GALILEO?

- Same core theme: **multi-turn dynamics** where correctness/safety depends on *dialogue history*, not just the current turn.
- Emphasizes **paired/controlled comparisons** (safe vs unsafe dialogues with minimal edits), similar in spirit to designing controlled multi-turn perturbations.
- Measures a **trade-off** (avoid bad behavior vs preserve helpfulness), analogous to robustness vs utility tensions.

## How is this different from GALILEO?

- Focus is **contextual safety** for **multimodal** assistants (image-grounded); not primarily truthfulness/robustness-to-misleading-followups per se.
- Uses an **LLM-judge discrete scoring** protocol (SA/Helpfulness 1–5) rather than time-to-failure / survival-style metrics.
- Benchmarks two specific intent-evolution patterns (escalation, context-switch), rather than broader robustness families (e.g., pressure, rebuttal, drift).

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO uses more transparent, automated metrics (e.g., turn-of-failure / survival) and strong control conditions, it may avoid some judge-subjectivity and enable sharper statistical analysis.

## Where GALILEO is weaker / needs to improve

- If GALILEO is text-only, it may miss the important class of failures where the *same dialogue* is safe/unsafe depending on visual context, and where safety hinges on intent tracking over turns.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adopting MTMCS-style **paired safe/unsafe minimal-edit trajectories** as a general recipe for multi-turn evaluation (even in text-only settings).
- [ ] Add a “**context-switch**” condition: explicitly unsafe framing at turn 1, followed by innocuous-sounding follow-ups; measure whether the model’s behavior remains conditioned on the initial intent.
- [ ] Add an “**escalation**” condition: benign early turns that gradually reveal harmful intent; measure delayed refusal vs overrefusal.
- [ ] If using judges, validate **judge stability** and optionally report agreement with human labels (they include this as an appendix-style check).

## Quotes / details to potentially cite

- Benchmark motivation: multi-turn contextual safety where “malicious intent can emerge gradually” and single-turn benchmarks miss this.
- Dataset size statement: “over 30 thousand multimodal (image+text) and unimodal (text-only) samples” and “12,032 dialogues.”
- Two setups: “escalation-based risk” and “context-switch risk.”
