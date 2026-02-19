# TurnBench-MS: A Benchmark for Evaluating Multi-Turn, Multi-Step Reasoning in Large Language Models

- Year: 2025
- Venue: arXiv
- Authors: Yiran Zhang, Mo Wang, Xiaoyang Li, Kaixuan Ren, Chencheng Zhu, Usman Naseem
- URL: https://arxiv.org/html/2506.01341v1
- BibTeX key (if we add it): zhang2025turnbenchms
- Tags: multi-turn; multi-step; interactive benchmark; rule discovery; robustness; process evaluation; contamination-resistance

## One-sentence takeaway

TurnBench evaluates multi-turn, multi-step reasoning via an interactive code-breaking game with feedback loops and (optionally) hidden verifier remapping, revealing a large human–LLM gap especially under the harder “Nightmare” setting.

## What problem does it solve?

- Existing reasoning benchmarks are often single-turn/single-shot and mainly score final answers, missing iterative hypothesis testing, feedback integration, and across-turn consistency.
- Static benchmarks risk pretraining contamination; TurnBench aims to be more resistant via combinatorial/dynamic game configurations.

## What is the core method / protocol?

- Interactive “Turing Machine” style deduction game:
  - Secret 3-digit code with digits 1–5.
  - Multiple “verifiers” each governed by one hidden active rule (HAC) from a rule pool.
  - Per round: propose a code → query up to 3 verifiers → receive binary PASS/FAIL feedback → decide to submit final code or SKIP and continue.
- Two modes:
  - Classic: verifier response corresponds to the queried verifier.
  - Nightmare: verifiers are secretly remapped; you query one but get feedback from another (mapping must be inferred).
- They also propose a *process-level* evaluation pipeline:
  - Extract the model’s inferred HACs from its reasoning text.
  - Compare inferred HACs to ground-truth HACs (categories like correct / incorrect / partially-contains).

## What are the key metrics?

- Final-task accuracy (solve the instance / find correct code), reported for Classic vs Nightmare.
- “Average turns” (efficiency) alongside accuracy.
- Intermediate/process evaluation: agreement between inferred HACs and ground-truth HACs (semantic match + partial credit categories).

## What are the main results?

- TurnBench contains 540 instances total: 270 Classic + 270 Nightmare (easy/medium/hard).
- Best reported model performance is much lower in Nightmare than Classic:
  - Classic: up to ~81.5% accuracy (best model in their study).
  - Nightmare: drops to ~17.8%.
- Human participants reportedly achieve 100% in both modes.

## How is this similar to GALILEO?

- Same underlying evaluation pressure: multi-turn robustness where the agent must update beliefs from feedback and remain consistent across rounds.
- Provides a concrete, interactive setting to measure “drift”/instability across turns (even if they frame it as reasoning rather than social persuasion).
- Includes a notion of *process-level* assessment, aligned with GALILEO’s interest in diagnosing *why* failures occur, not only final accuracy.

## How is this different from GALILEO?

- TurnBench is primarily a *reasoning/game* benchmark (rule discovery + deduction), not explicitly about social pressure, user persuasion, or sycophancy.
- Their intermediate-step evaluation depends on extracting conclusions from model-produced reasoning text (may be sensitive to prompting / refusal to reveal reasoning), whereas GALILEO may want behaviorally grounded signals.
- Nightmare mode introduces hidden verifier remapping (an adversarial latent-structure challenge) rather than conversational/adversarial user moves.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO measures robustness under *social* pressure (agreement, persuasion, user-provided “facts”), it targets a distinct—and under-benchmarked—failure mode not covered by TurnBench.
- GALILEO can potentially avoid reliance on chain-of-thought extraction by using externalized state/belief trackers or rubric-based turn-level judgments.

## Where GALILEO is weaker / needs to improve

- TurnBench demonstrates a clean, contamination-resistant interactive benchmark design; GALILEO should ensure similarly strong resistance (procedural generation, latent parameterization, or hidden mappings).
- The inclusion of ground-truth intermediate annotations is useful; GALILEO may need a comparable “step-level” supervision/evaluation signal (even if not CoT-based).

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a citation + brief comparison in related work under “interactive multi-turn reasoning benchmarks” (game-like feedback loops).
- [ ] Consider a “Nightmare-style” latent perturbation variant for GALILEO tasks (e.g., hidden remapping of tools/sources or partial observability) to stress-test belief update stability.
- [ ] If GALILEO uses intermediate belief states, report both final performance and turn-level/process metrics (e.g., belief consistency, revision correctness) analogous to TurnBench’s HAC correctness.

## Quotes / details to potentially cite

- Abstract framing: benchmarks are often single-turn; TurnBench uses an interactive code-breaking task with sequential guesses and structured feedback.
- Reported gap: best model ~81.5% (Classic) vs ~17.8% (Nightmare), while humans 100% in both.
- Two-mode design: Classic vs Nightmare with hidden verifier remapping.
