# Multi-turn Jailbreaking Attack in Multi-Modal Large Language Models

- Year: 2026
- Venue: arXiv
- Authors: Badhan Chandra Das; Md Tasnim Jawad; Joaquin Molto; M. Hadi Amini; Yanzhao Wu
- URL: https://arxiv.org/abs/2601.05339
- BibTeX key (if we add it): das2026multiturnjailbreak
- Tags: multi-turn, jailbreak, multimodal, typography, defense, response-filtering

## One-sentence takeaway

Introduces a 3-turn multi-modal jailbreaking recipe (benign → hypothetical framing → harmful ask seeded via typographic image) and a training-free defense (FragGuard) that fragments outputs and uses multiple LLMs to conservatively score/suppress harmfulness.

## What problem does it solve?

- Prior MLLM jailbreaking work focuses on single-turn or prompt-only attacks; multi-turn conversational escalation for MLLMs is less systematically analyzed.
- Need defenses that are (a) training-free, (b) usable for black-box / closed models, and (c) robust under multi-turn adversarial interactions.

## What is the core method / protocol?

- **Attack (multi-turn, black-box):**
  - Uses a **typography-manipulated image** that hides/embeds a forbidden request as caption-like text.
  - Turn 1: benign query (e.g., describe the image).
  - Turn 2: ask for **hypothetical/creative elaboration** (e.g., a movie script) to move the model into a permissive narrative mode.
  - Turn 3: request the model to comply with the harmful request embedded in the image.
  - Claim: step-wise escalation increases attack success vs direct harmful ask.

- **Defense (FragGuard; training-free output moderation):**
  - **Fragment** the model response into fixed-length token chunks.
  - Score each fragment’s harmfulness/toxicity with **multiple LLM “judges”** (paper lists OpenAI o1, Gemini-2.5-Flash-lite, and LLaMA-3 70B).
  - Aggregate conservatively: take the **maximum** score across fragments and judges.
  - If above threshold, suppress/replace with a safe refusal.

## What are the key metrics?

- Attack success rate / harmful compliance rate under the multi-turn protocol (paper describes “severity” and harmfulness levels).
- Defense effectiveness: reduction in successful jailbreaking / harmful outputs after FragGuard.
- (Likely) false positives / utility impact from suppression (not confirmed from skim).

## What are the main results?

- Demonstrates that multi-turn interaction + typographic image prompt can elicit harmful outputs from a range of open and closed MLLMs.
- FragGuard (multi-LLM, fragment-based scoring) mitigates attack-generated harmful responses without model fine-tuning.

## How is this similar to GALILEO?

- Shares the theme of **adversarial evaluation** and **systematic safety testing** of (multi-modal) models.
- Uses a **pipeline framing**: attack generation + automated evaluation/guarding (multi-judge scoring resembles ensemble evaluation).

## How is this different from GALILEO?

- Focuses on a specific jailbreak recipe + output-filter defense, rather than (presumably) broader capability evaluation or task-level benchmarking.
- Relies on **LLM-based toxicity judging** and thresholding; not necessarily grounded in task-specific correctness/robustness metrics.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO uses **task-grounded, reproducible metrics** (vs judge toxicity), it may be less sensitive to judge-model drift and prompt variance.
- If GALILEO emphasizes **threat-model clarity** and controlled ablations, it may be easier to interpret than max-over-judges heuristics.

## Where GALILEO is weaker / needs to improve

- If GALILEO does not yet include **multi-turn multi-modal attack scripts**, this paper is a reminder that multi-turn escalation is a realistic failure mode.
- Might need explicit **conversation-level** evaluation (not just single prompt-response pairs).

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a “multi-turn escalation” track for MLLM red-teaming (benign grounding → hypothetical framing → unsafe request).
- [ ] Consider output-level **fragmentation** as an analysis tool: where in the response harmfulness appears (early vs late) and whether guardrails fail locally.
- [ ] If using LLM judges, document robust aggregation (e.g., max vs mean) and measure **utility loss** (false refusals) under benign tasks.

## Quotes / details to potentially cite

- Paper frames contributions as: (1) a novel **multi-turn jailbreaking attack** for MLLMs, (2) a training-free defense (**FragGuard**) using **fragment-optimized multi-LLM** harmfulness measurement, and (3) extensive evaluation on SOTA open/closed MLLMs.
- Attack sketch (3 turns): benign image description → hypothetical scenario generation → comply with typographically embedded harmful request.
