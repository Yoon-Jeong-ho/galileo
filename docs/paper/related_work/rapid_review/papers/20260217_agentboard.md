# AgentBoard: An Analytical Evaluation Board of Multi-turn LLM Agents

- Year: 2024
- Venue: NeurIPS 2024 (Oral)
- Authors: Chang Ma; Junlei Zhang; Zhihao Zhu; Cheng Yang; Yujiu Yang; Yaohui Jin; Zhenzhong Lan; Lingpeng Kong; Junxian He
- URL: https://arxiv.org/abs/2401.13178
- BibTeX key (if we add it): agentboard_2024
- Tags: agents, multi-turn, evaluation, benchmark

## One-sentence takeaway

AgentBoard is a multi-task benchmark + evaluation toolkit for *analytical* (process-sensitive) evaluation of LLM agents in partially observable, multi-round environments, emphasizing a fine-grained “progress rate” rather than only final success.

## What problem does it solve?

- Existing agent benchmarks are hard to compare across heterogeneous environments and often reduce outcomes to a single terminal success metric.
- Terminal success alone hides *how* agents fail (e.g., making partial progress, getting stuck, losing grounding) and gives limited interpretability of multi-turn behavior.
- Multi-round, partially observable settings are especially difficult to standardize and analyze.

## What is the core method / protocol?

- Curates a “board” of diverse goal-oriented tasks/environments (GitHub README: 9 distinct tasks) designed for multi-round interaction and partial observability.
- Provides an open-source framework to run a (simple reflex) LLM agent across tasks with standardized logging.
- Adds an “analytical evaluation” layer: a toolkit/panel (W&B-based) to break down performance across dimensions (progress tracking, grounding accuracy, long-range interactions, sub-skills, trajectories).

## What are the key metrics?

- **Progress rate** (fine-grained): measures incremental advancement toward task completion during interaction, to distinguish “made meaningful progress but failed at the end” vs “no real progress.”
- **Final success rate** (implied baseline comparison).
- **Grounding accuracy** (as an analysis axis/toolkit output; per README).
- Additional breakdown slices: hard vs easy examples, long-range interaction behavior, sub-skill performance (framework/tooling).

## What are the main results?

- The paper positions progress-rate analysis as revealing differences obscured by terminal success-only reporting (claimed in abstract and NeurIPS dataset/benchmark track description).
- Primary contribution is the benchmark + standardized evaluation + analysis tooling; empirical results compare multiple proprietary/open models (per README) under a unified setup.

## How is this similar to GALILEO?

- Shared motivation: **multi-turn evaluation** where process matters (not just endpoint accuracy).
- Emphasizes **trajectory-aware** analysis and diagnostics of failure modes over long interactions.

## How is this different from GALILEO?

- AgentBoard targets **generalist agent task completion** (tool/environment interaction) rather than GALILEO’s focus on **robustness under conversational pressure / drift / sycophancy-like failures**.
- Their metrics focus on goal progress in environments; GALILEO’s core measurements center on *robustness and stability* under structured perturbations (e.g., persuasion/pressure, multi-turn degradation, time-to-failure style signals).

## Where GALILEO is stronger / cleaner (if true)

- Cleaner alignment to our paper’s thesis if we frame GALILEO around **robustness/stability metrics** (e.g., turn-of-failure / survival-style) rather than broad agent capability.
- More targeted to *social/argumentative pressure* and *belief/answer stability* phenomena (AgentBoard is broader but less specific).

## Where GALILEO is weaker / needs to improve

- AgentBoard’s emphasis on **standardized multi-environment agent evaluation** and an **analysis dashboard** suggests we could strengthen our “tooling + interpretability” story (especially interactive breakdowns and trajectory visualization).
- They explicitly operationalize partial observability and exploration; if GALILEO claims general multi-turn robustness, we may need clearer discussion of environment/state observability assumptions.

## Action items for GALILEO (experiments / method / writing)

- [ ] In related work, cite AgentBoard as prior work advocating **process-sensitive** metrics for multi-turn agent evaluation (progress-rate vs terminal success).
- [ ] Consider adding a small “analysis views” subsection: examples of failure-mode slices (hard/easy; long-range) and a lightweight dashboard/log schema.
- [ ] Clarify positioning: GALILEO is *not* a general agent task suite; it is a robustness/stability evaluation (to avoid scope creep).

## Quotes / details to potentially cite

- Abstract: proposes a “fine-grained progress rate metric that captures incremental advancements” and criticizes success-rate-only evaluation.
- GitHub README (project description): stresses multi-round interaction, partial observability, and analysis axes including “fine-grained progress rates” and “grounding accuracy,” with W&B-based visualization.
