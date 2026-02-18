# MURPHY: Multi-Turn GRPO for Self-Correcting Code Generation

- Year: 2025 (arXiv v1 Nov 2025; v2 Feb 2026)
- Venue: arXiv (submitted to ICML)
- Authors: Vijay Lingam; Sujay Sanghavi; Behrooz Omidvar-Tehrani; Jun Huan; Anoop Deoras; Stefano Soatto
- URL: https://arxiv.org/abs/2511.07833
- BibTeX key (if we add it): murphyMultiTurnGRPO2025
- Tags: multi-turn, RLVR, GRPO, execution-feedback, code-generation, self-correction, agentic

## One-sentence takeaway

Extends GRPO-style RL with verifiable rewards from single-turn prompts to **multi-turn code-agent trajectories**, propagating success backward across iterative fix attempts using execution feedback + pruning to keep training tractable.

## What problem does it solve?

- Standard RLVR/GRPO post-training assumes a **single prompt → single response → reward** setup.
- Agentic code generation often needs **multiple iterations**: write code, run tests, read errors, revise, repeat.
- Single-turn RLVR can improve “first try” but may fail to train **self-correction behavior across turns**.

## What is the core method / protocol?

- Proposes *MURPHY*, a multi-turn RLVR framework that:
  - builds a **feedback-conditioned rollout tree**: from an initial prompt, sample a group of solutions; for failures, **execute** and append feedback (errors/tests) to the context; resample refinements for several turns.
  - performs **trajectory-level credit assignment**: rewards achieved in later turns are propagated to earlier-turn decisions so “failed but improvable” attempts that lead to eventual success get positive learning signal.
  - uses **pruning** to bound compute: keep only “promising” trajectories/branches so multi-turn optimization cost does not blow up.
- Positions itself as a minimal extension of GRPO to multi-turn interaction, avoiding auxiliary learned critics/verifiers.

## What are the key metrics?

- Code generation success metrics (reported as):
  - pass@1 (and multi-iteration performance, i.e., success after several refine-execute turns)
  - comparisons under **compute-matched** GRPO baselines

## What are the main results?

- Across two model families (Qwen, OLMo) and multiple coding benchmarks, MURPHY improves multi-iteration performance.
- Reports up to **~8% absolute pass@1 gain** over compute-matched GRPO baselines.
- Claims to outperform prior leading method incorporating multi-turn execution feedback (μCode / related baselines).

## How is this similar to GALILEO?

- Both care about **multi-turn dynamics** (not just single-turn accuracy) and how feedback/pressure signals change behavior over time.
- The paper’s emphasis on “credit assignment over trajectories” is conceptually aligned with GALILEO’s need to summarize and analyze **turn-by-turn failure/recovery** patterns (even though GALILEO is primarily evaluation).

## How is this different from GALILEO?

- Domain: code generation with **verifiable execution feedback**; GALILEO targets conversational pressure / belief drift vs evidence-driven revision.
- Goal: training algorithm (post-training RLVR) vs GALILEO’s benchmarking/measurement + protocol design.
- Feedback type: hard executor signals (tests/errors) vs social/argumentative pressure signals (often non-verifiable).

## Where GALILEO is stronger / cleaner (if true)

- GALILEO’s core contribution is disentangling **pressure-driven drift** vs **evidence-driven revision** with controls; MURPHY largely treats feedback as “useful correction signal” (execution) and does not address social-pressure confounds.
- GALILEO can evaluate models without requiring expensive RL fine-tuning or tool execution during training.

## Where GALILEO is weaker / needs to improve

- If GALILEO eventually wants mitigation baselines, MURPHY highlights that **multi-turn objectives** matter: training solely for “turn-0 correctness” can miss the behavior we measure (recovery, stability, calibrated updates).

## Action items for GALILEO (experiments / method / writing)

- [ ] When discussing mitigation baselines, explicitly distinguish **single-turn** vs **multi-turn** optimization targets; note that multi-turn credit assignment is a known missing piece in GRPO-style RLVR.
- [ ] Consider a “multi-turn objective” section in related work: execution-feedback RLVR (code) as an existence proof that multi-turn RL training is feasible with pruning.
- [ ] If we propose any training-time interventions, cite MURPHY as a template: rollout trees + pruning + trajectory-level assignment; then discuss what is hard about translating to GALILEO (lack of verifiable reward).

## Quotes / details to potentially cite

- Abstract-level summary: MURPHY “extends GRPO to optimize over multi-turn trajectories where models iteratively refine solutions” and “incorporates execution feedback directly into training,” combining a “feedback-conditioned rollout tree” with “trajectory-level credit assignment,” using pruning to reduce cost.
- Motivation framing: GRPO “fundamentally designed for a single-turn setting… with no notion of multi-turn interaction defined in their objective” (intro framing from the HTML).
