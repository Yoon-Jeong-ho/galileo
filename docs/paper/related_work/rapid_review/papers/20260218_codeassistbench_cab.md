# CodeAssistBench (CAB): Dataset & Benchmarking for Multi-turn Chat-Based Code Assistance

- Year: 2025
- Venue: NeurIPS 2025 (Datasets and Benchmarks Track)
- Authors: Myeongsoo Kim; Shweta Garg; Baishakhi Ray; Varun Kumar; Anoop Deoras
- URL: https://arxiv.org/abs/2507.10646
- BibTeX key (if we add it): codeassistbench2025kim
- Tags: multi-turn, benchmark, code-assistance, project-grounded, github-issues, containers, evaluation

## One-sentence takeaway

CAB is an automated benchmark that turns real GitHub “question” issues into multi-turn, containerized, project-grounded assistance tasks, revealing a large performance gap vs. StackOverflow-style single-turn Q&A.

## What problem does it solve?

- Existing programming-assistant benchmarks often:
  - focus on single-turn code generation or isolated snippets,
  - lack full-project/environment grounding,
  - require substantial manual curation,
  - under-measure the iterative clarification loops that dominate real help-seeking.
- CAB targets *multi-turn* assistance where success depends on understanding a specific repo + environment and guiding the user through clarifications.

## What is the core method / protocol?

- **Dataset construction (fully automated)** from GitHub issues labeled/tagged as questions:
  - Repository selection (filters like stars, creation date, permissive licenses; then a “community score” based on closed question/help issues).
  - Issue filtering:
    - regex-based rules (e.g., multiple participants; remove media-heavy issues),
    - LLM-based filters for quality/safety/reproducibility and to remove low-value comments.
  - Turn reconstruction: group consecutive author/maintainer messages into turns.
  - **Build environment generation**: auto-generate and validate Docker build scripts/configs using an LLM and repo artifacts.
  - **Satisfaction conditions**: LLM extracts explicit “done when …” criteria from the original issue thread.
  - **Simulated user follow-ups**: retrieve similar historical maintainer→user pairs (BM25) to guide realistic user responses.
- **Evaluation framework** (multi-agent simulation):
  - User agent (poses issue + follow-ups),
  - Maintainer agent (the model under test; can run commands in container),
  - Judge agent (LLM-based grading against satisfaction conditions + interaction quality).
  - Conversations end when user is satisfied or at a max turn limit (reported as configurable; examples mention up to 10 turns).

## What are the key metrics?

- Primary outcome is essentially **task success / accuracy** on issues (meeting extracted satisfaction conditions), with execution success a hard requirement when a Docker environment is involved.
- Judge dimensions described: technical correctness, satisfaction completeness, and interaction quality (conciseness/helpfulness).

## What are the main results?

- Scale: **3,286** real-world issues from **214** repositories across **7** languages.
- Strong models do well on StackOverflow-style Q&A (**~70–83%** accuracy), but on CAB (especially post-training-cutoff repos) they solve only **~7.22–16.49%** of issues (paper highlights ~16.49% on the post-cutoff split).
- Core claim: today’s LLMs still struggle with realistic, project-specific, multi-turn assistance despite looking strong on traditional benchmarks.

## How is this similar to GALILEO?

- Shared emphasis on **multi-turn evaluation** rather than single-shot metrics.
- Highlights that realistic performance gaps appear when you:
  - add *context reconstruction* and *interaction*,
  - evaluate *trajectories* rather than isolated answers,
  - stress-test models on *distribution shift* (post-cutoff repositories).

## How is this different from GALILEO?

- CAB is about **programming assistance** (codebases + containers + execution), not belief revision / drift under social pressure.
- Their “multi-turn” is task-clarification and troubleshooting; GALILEO’s core concerns (robustness under pressure, sycophancy/persuasion dynamics, belief stability) are mostly orthogonal.
- Evaluation relies on an **LLM judge** + extracted satisfaction conditions; GALILEO likely cares about *stability signals* and *drift vs. evidence-driven revision controls* more directly.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO provides explicit **controls for drift vs. revision** and pressure/sycophancy manipulations, it covers a failure mode CAB does not target.
- CAB’s success definition is tied to “issue resolved” satisfaction conditions; GALILEO can be cleaner on *truth-tracking / epistemic stability* objectives.

## Where GALILEO is weaker / needs to improve

- CAB is a strong example of **scalable, automated dataset generation** + **grounded evaluation harness**; GALILEO could borrow:
  - automated scenario harvesting pipelines,
  - post-cutoff splits for continual difficulty,
  - executable environments as hard oracles where applicable.

## Action items for GALILEO (experiments / method / writing)

- [ ] In related work, cite CAB as evidence that **multi-turn, grounded settings expose large capability gaps** even when single-turn benchmarks saturate.
- [ ] Consider a “post-cutoff” style split (or analogous *time-based shift*) for GALILEO scenarios to reduce benchmark overfitting.
- [ ] If feasible, define “satisfaction conditions” analogs for GALILEO (explicit success criteria per dialogue) to improve judge reliability / reproducibility.
- [ ] Add a short paragraph contrasting CAB-style multi-turn troubleshooting with GALILEO’s multi-turn pressure/stability axis.

## Quotes / details to potentially cite

- “CAB … the first benchmark for evaluating multi-turn, project-grounded programming assistance at scale.”
- Automatic pipeline: filters noise, extracts runnable contexts, builds executable containers, and verifies environment correctness.
- Scale and gap: 3,286 issues / 214 repos / 7 languages; ~70–83% on StackOverflow-style vs ~7.22–16.49% on CAB post-cutoff repos.
- Evaluation architecture: user agent + maintainer agent + judge agent; grade against extracted satisfaction conditions; execution success required when Docker env used.
