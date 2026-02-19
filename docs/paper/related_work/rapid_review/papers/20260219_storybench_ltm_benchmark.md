# StoryBench: A Dynamic Benchmark for Evaluating Long-Term Memory with Multi Turns

- Year: 2025
- Venue: arXiv
- Authors: Luanbo Wan, Weizhi Ma
- URL: https://arxiv.org/html/2506.13356v1
- BibTeX key (if we add it): storybench-wan-2025
- Tags: long-term-memory, multi-turn, benchmark, interactive-fiction, sequential-reasoning, self-correction

## One-sentence takeaway

StoryBench evaluates LLM long-term memory via branching interactive-fiction trajectories, explicitly stressing sequential decision dependencies and a “self-recovery” mode that requires backtracking to earlier mistakes.

## What problem does it solve?

- Existing long-context / LTM benchmarks skew toward static QA or retrieval-like recall and under-test (a) *dynamic* stateful multi-turn interactions, (b) sequential causal dependencies across decisions, and (c) flexibility (multiple valid solution paths).
- Benchmarking LTM is confounded when tasks don’t force models to *use* earlier information to make future decisions in an evolving environment.

## What is the core method / protocol?

- Benchmark framing: an LLM plays through interactive fiction with *branching storylines* (decision tree / DAG).
- Two evaluation modes:
  - **Immediate Feedback**: after an incorrect choice, the system tells the model it was wrong and prompts it to retry until it selects the correct option (tests short-horizon adjustment).
  - **Self Recovery**: no immediate feedback; wrong choices can propagate to failure endings; after failure the model must identify the earliest wrong decision and attempt recovery from there (tests long-horizon causal tracing / self-correction).
- Dataset: constructed from an interactive fiction game (“The Invisible Guardian”), transcribed/annotated into scene nodes + choice nodes (reported: 311 scene nodes, 86 choice nodes).
- Evaluation: run models on many branching paths (paper claims 80+ paths), repeat trials for robustness, compute decision-level and trajectory-level metrics.

## What are the key metrics?

- Decision accuracy metrics:
  - Overall accuracy across decisions
  - First-try accuracy
  - Longest consecutive correct sequence
  - Easy vs hard accuracy (hard requires distant recall / latent state tracking / multi-step reasoning)
- Interaction difficulty / failure metrics:
  - Retry count (Immediate Feedback)
  - Max error per choice / thresholded error count (paper mentions threshold like 9)
- Efficiency metrics (auxiliary): runtime cost, token consumption
- Trajectory outcome metric: success count / task completion (finishing paths)

## What are the main results?

- Closed models (reported: GPT-4o, Claude 3.5 Sonnet) outperform others on completion/self-recovery stability, but **all** struggle with robust self-recovery and consistently revising earlier mistakes.
- Performance drops substantially in Self Recovery vs Immediate Feedback, highlighting that many models rely on short-horizon feedback to correct mistakes.
- Failure analysis highlights:
  - **Contextual inconsistency** (contradicting earlier events/entities/motivations) as a knowledge-retention failure.
  - **Shallow backtracking** (only revising 1–2 steps) as a sequential-reasoning failure when errors arise from longer causal chains.

## How is this similar to GALILEO?

- Shares the core concern: evaluating (and ultimately improving) *long-term memory* in multi-turn settings where earlier context must influence later decisions.
- Emphasizes *dynamic* interactions rather than static single-shot tasks, aligning with agent-style evaluation motivations.

## How is this different from GALILEO?

- StoryBench is primarily a **benchmark/dataset + evaluation protocol**, not a memory architecture or agent system.
- Uses interactive fiction game trajectories as the task substrate; GALILEO may target broader tool-using / real-world tasks (depending on the paper’s scope).
- Includes explicit “self-recovery” backtracking evaluation mode as a centerpiece, which may or may not be present in current GALILEO evaluations.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO evaluates memory in tool-use or open-world tasks, it may better reflect real deployment constraints than interactive fiction.
- GALILEO could integrate memory with retrieval/structured state in a more controlled way than purely narrative branching.

## Where GALILEO is weaker / needs to improve

- If GALILEO lacks a **no-feedback self-repair/backtracking** evaluation, StoryBench suggests a concrete gap: testing whether agents can diagnose earlier decisions after delayed failure.
- If GALILEO metrics are mostly QA-style (accuracy) rather than *trajectory completion*, StoryBench argues completion is a more meaningful stress test.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a “self-recovery” evaluation variant for at least one GALILEO task: allow the agent to run until failure with no hints; then require it to propose the earliest incorrect action/state update and re-run from there.
- [ ] Track *completion/success count* alongside local accuracy; report gap between easy vs hard decision points to isolate sequential reasoning failures.
- [ ] Consider building a small branching-task suite (doesn’t have to be fiction) where actions have delayed consequences, to test long-range causal tracing.

## Quotes / details to potentially cite

- Benchmark design motivation: “interactive fiction games, featuring dynamically branching storylines with complex reasoning structures… hierarchical decision trees, where each choice triggers cascading dependencies across multi-turn interactions.”
- Two modes: “Immediate Feedback… [and] Self Recovery… requiring the model to identify and revise past decisions on its own.”
- Dataset size/structure (as reported): “311 scene nodes and 86 choice nodes” organized as a DAG with scene/choice nodes.
