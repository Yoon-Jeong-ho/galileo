# NL2Repo-Bench: Towards Long-Horizon Repository Generation Evaluation of Coding Agents

- Year: 2025
- Venue: arXiv
- Authors: Jingzhe Ding et al.
- URL: https://arxiv.org/abs/2512.12730
- BibTeX key (if we add it): NL2RepoBench2025
- Tags: agents, coding, long-horizon, benchmark, robustness

## One-sentence takeaway

NL2Repo-Bench evaluates whether coding agents can generate a complete, installable multi-module Python repo from a single requirements doc, finding that long-horizon repo generation is still largely unsolved (<40% avg test pass rates for top agents).

## What problem does it solve?

- Existing coding benchmarks over-emphasize short-horizon tasks (single-function generation, local repair) and under-measure end-to-end “build a repo from scratch” competence.
- Lack of *verifiable* evaluation for sustained planning/execution over many steps and files.

## What is the core method / protocol?

- Benchmark task: given (1) a natural-language requirements document and (2) an empty workspace, an agent must:
  - design repo architecture,
  - manage dependencies,
  - implement multi-module logic,
  - produce an installable Python library,
  - and pass unit tests (implied by the “test pass rate” reporting).
- Reports aggregate performance and analyzes qualitative failure modes observed in long trajectories.

## What are the key metrics?

- Average test pass rate (primary reported outcome in abstract).
- Completion success / “entire repository correctly completed” frequency (qualitative/secondary).
- Failure mode incidence (premature termination, coherence loss, dependency fragility, inadequate planning).

## What are the main results?

- Strongest evaluated agents/models achieve **below 40% average test pass rates**.
- Agents **rarely complete an entire repository correctly**.
- Common long-horizon failures include:
  - premature termination,
  - loss of global coherence across files,
  - fragile cross-file dependencies,
  - insufficient planning across hundreds of interaction steps.

## How is this similar to GALILEO?

- Shared focus on *long-horizon* behavior where errors compound over time (state drift, coherence breaks, premature stopping).
- Emphasizes evaluation protocols that expose failures not visible in short, local tasks.
- Provides a concrete taxonomy of trajectory-level failure modes that may align with GALILEO’s concerns about robustness over extended interactions.

## How is this different from GALILEO?

- Domain is **software repository generation** (coding-agent autonomy) rather than conversational robustness / belief drift / social pressure (if that is GALILEO’s primary focus).
- Outcome metric is **unit-test pass rate** and repo correctness, not behavioral/epistemic stability metrics.
- Task setup starts from a requirements document + empty workspace, not from multi-turn conversational prompts.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO targets *controlled perturbations and stability metrics* in multi-turn settings, it may offer more precise causal isolation than a broad repo-generation task.
- GALILEO-style analyses could yield more fine-grained “turn-of-failure / time-to-failure” style curves than a single end-of-run pass rate.

## Where GALILEO is weaker / needs to improve

- If GALILEO does not include end-to-end “build a complex artifact” tasks, it may under-test cross-file/global-coherence failure modes that show up in NL2Repo-Bench.
- May need stronger measurement of *premature termination* and *global plan consistency* across long tool-using trajectories.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add/mention a *trajectory-level* failure taxonomy section (premature termination, global coherence loss, cross-file dependency fragility) as analogous long-horizon failure modes.
- [ ] Consider a small “artifact construction” stress test (even non-coding) to probe coherence across many steps/modules.
- [ ] When discussing long-horizon evaluation, cite NL2Repo-Bench as evidence that long-horizon agentic competence remains a bottleneck even for strong models.

## Quotes / details to potentially cite

- “Given only a single natural-language requirements document and an empty workspace, agents must autonomously design the architecture, manage dependencies, implement multi-module logic, and produce a fully installable Python library.”
- “Even the strongest agents achieve below 40% average test pass rates and rarely complete an entire repository correctly.”
- Failure modes highlighted: “premature termination, loss of global coherence, fragile cross-file dependencies, and inadequate planning over hundreds of interaction steps.”
