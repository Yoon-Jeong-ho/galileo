# StoryBench: A Dynamic Benchmark for Evaluating Long-Term Memory with Multi Turns

- Year: 2025
- Venue: arXiv
- Authors: Luanbo Wan, Weizhi Ma
- URL: https://arxiv.org/abs/2506.13356
- BibTeX key (if we add it): wan2025storybench
- Tags: multi-turn, long-term memory, benchmark, interactive-fiction, branching, recovery

## One-sentence takeaway

StoryBench evaluates LLM long-term memory via branching interactive-fiction trajectories, including a harder “self-recovery” mode where models must diagnose and revise earlier wrong decisions after reaching a failure ending.

## What problem does it solve?

- Lack of a standardized, *dynamic* benchmark for long-term memory that goes beyond long-context recall to test (i) knowledge retention across many turns and (ii) sequential/causal reasoning where earlier choices create downstream dependencies.
- Existing LTM benchmarks are argued to be limited in flexibility and in capturing “dynamic sequential reasoning” under evolving state.

## What is the core method / protocol?

- Environment: interactive fiction with **branching storylines** (hierarchical decision trees).
- Per turn, the model receives: scene description/dialogue + multiple action options, then must choose an action.
- Two evaluation settings:
  - **Immediate Feedback**: wrong choices are flagged immediately (allows quick correction).
  - **Self Recovery**: the story continues without hints to a failure ending; the model must **trace back** and revise earlier decisions to recover.
- Dataset: an annotated interactive-fiction dataset with “cohesive narrative continuity”, branching, and multi-solution mechanisms (as claimed).

## What are the key metrics?

(From the paper’s framing; details likely include per-path/trajectory aggregation.)

- Correct decision rate (turn-level accuracy / choice correctness)
- Task success counts (e.g., successful completion of story paths)
- Performance split by mode (Immediate Feedback vs Self Recovery)

## What are the main results?

- Evaluated 4 advanced LLMs on 80+ branching story paths.
- GPT-4o and Claude 3.5 Sonnet are reported as stronger on retention/sequential reasoning than other tested models, but **all models struggle in Self Recovery** (difficulty revising earlier mistakes without feedback).
- The benchmark is claimed to be robust/reliable via repeated trials and to enable granular failure analysis.

## How is this similar to GALILEO?

- Shared emphasis on **multi-turn evaluation** where earlier turns causally influence later success/failure (trajectory-level outcomes).
- The **Self Recovery** mode is conceptually adjacent to measuring *recovery after a failure* (i.e., not only whether a model fails, but whether it can diagnose and correct).
- Provides benchmark-design language for separating “knowledge retention” vs “sequential reasoning” dimensions (useful for positioning GALILEO’s evaluation axes).

## How is this different from GALILEO?

- StoryBench focuses on narrative interactive-fiction decision making (LTM + sequential reasoning), not primarily on **social pressure / persuasion / drift-vs-revision controls**.
- “Failure” is defined as reaching a wrong branch / failure ending; it is not necessarily aligned with *belief flip* or *stance drift* under user pressure.
- Metrics appear to emphasize correctness/success counts rather than time-to-event / survival-style reporting (unless present later in the paper).

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO’s core claim is about pressure-induced drift and recovery dynamics, it likely has cleaner **paired-control** constructions (pressure vs evidence) and more targeted “flip/recovery” metrics than narrative correctness.

## Where GALILEO is weaker / needs to improve

- StoryBench’s “dynamic environment” framing (branching dependencies) is a strong example of **stateful evaluation**; if GALILEO currently uses mostly scripted dialogue variants, adding more explicit branching/state could strengthen ecological validity.
- Self-recovery without immediate feedback is a good stressor; GALILEO could benefit from a comparable “no-hints until failure” slice (if not already present).

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding a **no-immediate-feedback** evaluation mode (let the interaction run to a “failure” state, then measure whether the model can backtrack / revise earlier commitments).
- [ ] If applicable, add a benchmark-design paragraph contrasting **static QA** vs **dynamic branching trajectories** to motivate GALILEO’s multi-turn setup.
- [ ] Consider reporting separate axes for (i) retention of earlier facts/claims and (ii) sequential dependency reasoning, even if the content domain differs.

## Quotes / details to potentially cite

- “We propose a novel benchmark framework based on interactive fiction games, featuring dynamically branching storylines with complex reasoning structures.”
- Two modes: “Immediate Feedback” vs “Self Recovery” where models “independently trace back and revise earlier choices after failure.”
