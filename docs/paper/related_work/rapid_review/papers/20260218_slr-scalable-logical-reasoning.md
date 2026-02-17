# SLR: Automated Synthesis for Scalable Logical Reasoning

- Year: 2025
- Venue: arXiv
- Authors: Lukas Helff, Ahmad Omar, Felix Friedrich, Antonia Wüst, Hikaru Shindo, Tim Woydt, Rupert Mitchell, Patrick Schramowski, Wolfgang Stammer, Kristian Kersting
- URL: https://arxiv.org/abs/2506.15787
- BibTeX key (if we add it): SLR2025Helff
- Tags: logic, benchmark, inductive-reasoning, symbolic-judge, curriculum

## One-sentence takeaway

SLR is an automated pipeline that *synthesizes* inductive-logic tasks together with an executable validator (“symbolic judge”), enabling scalable benchmarking (SLR-Bench) and curriculum training without human labels.

## What problem does it solve?

- Existing “reasoning” benchmarks often:
  - focus on deduction / multiple-choice formats,
  - use LLM judges,
  - risk contamination/memorization as items resemble pretraining data,
  - lack controllable difficulty and *verifiable* evaluation.
- Inductive logical reasoning (inferring a general rule from examples) is under-evaluated and hard to score reliably.

## What is the core method / protocol?

- Input: a user-defined task specification (vocabulary/grammar + task config).
- A task synthesizer automatically generates, per instance:
  1) an instruction prompt for an inductive reasoning task,
  2) a **latent ground-truth rule** (the target hypothesis),
  3) a **validation program** that deterministically checks candidate rules / outputs.
- This yields:
  - **SLR-Bench**: ~19k prompts organized into **20 curriculum levels** spanning relational, arithmetic, and recursive complexity.
- Training modes:
  - SFT on synthesized tasks, and/or
  - RL using reward from the symbolic judge (verifiable feedback).

## What are the key metrics?

- Task accuracy on SLR-Bench across curriculum levels (difficulty scaling).
- Cost / test-time compute (inference tokens; reported dollar cost for 1k prompts for heavy “reasoning” models).
- Generalization performance on downstream reasoning benchmarks (the paper mentions e.g., GPQA and CLUTRR).

## What are the main results?

- Many LLMs can produce *syntactically valid* rules but still fail **logical inference correctness**, especially as complexity increases.
- “Reasoning” LLMs improve accuracy but can be extremely expensive at test time (reported >$300 per 1,000 prompts in their setting).
- Curriculum learning with SLR roughly **doubles** Llama-3-8B accuracy on SLR-Bench; they claim parity with Gemini-Flash-Thinking at much lower compute.
- Improvements transfer to other benchmarks (claimed broad generalization).

## How is this similar to GALILEO?

- Shared theme: *evaluation protocols that stress systematic generalization* rather than surface heuristics.
- Uses a verifier/judge with programmatic checks (relevant if GALILEO uses structured evaluation/metrics to avoid judge brittleness).
- Curriculum and difficulty control echoes “controlled perturbations” style evaluation design.

## How is this different from GALILEO?

- SLR targets **single-task inductive logical reasoning** (rule induction) more than multi-turn interaction dynamics.
- Focuses on synthetic logic-task generation + symbolic validation, not conversational robustness/failure modes across turns.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO centers on multi-turn robustness or behavioral stability, it likely captures ecological interaction effects that SLR’s single-instance logic tasks do not.

## Where GALILEO is weaker / needs to improve

- If GALILEO lacks a fully programmatic judge for some dimensions, SLR is a good reference for “verifiable reward” design.
- If GALILEO lacks controllable curricula, SLR’s 20-level organization is a useful pattern.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding a **symbolic/programmable validator** for at least one GALILEO subtask to reduce evaluator ambiguity.
- [ ] Consider reporting **cost vs accuracy** tradeoffs explicitly (SLR highlights how test-time compute can dominate).
- [ ] If applicable, add a curriculum / tiered difficulty schedule rather than a flat benchmark.

## Quotes / details to potentially cite

- “Given a user’s task specification, SLR automatically synthesizes (i) an instruction prompt … (ii) a validation program … and (iii) the latent ground-truth rule.” (abstract)
- “SLR-Bench … 19k prompts … 20 curriculum levels … relational, arithmetic, and recursive complexity.” (abstract)
- “Recent reasoning LLMs … costs exceeding $300 for just 1,000 prompts.” (abstract)
