# User-Oriented Multi-Turn Dialogue Generation with Tool Use at scale

- Year: 2026
- Venue: arXiv
- Authors: Jungho Cho; Minbyul Jeong; Sungrae Park
- URL: https://arxiv.org/abs/2601.08225
- BibTeX key (if we add it): cho2026useroriented
- Tags: multi-turn, tool-use, data-generation, robustness

## One-sentence takeaway

A synthetic-data pipeline for *long*, realistic multi-turn tool-use dialogues by separating (i) task/tool synthesis from (ii) a rule-based “user simulator” that drip-feeds subtasks and feedback.

## What problem does it solve?

- Current tool-use / agent datasets often assume **static toolsets** and/or **single-shot, efficiency-biased** interactions.
- When an LRM simulator is tasked with “solve the task,” it tends to finish in **minimal turns**, under-producing clarifications, incremental requests, and feedback loops that characterize real human–agent use.

## What is the core method / protocol?

- Generate tool-use training data via simulation with three modular components:
  - **Dynamic Tool & Task Synthesis:** an LRM synthesizes domain-specific tools (incl. DB/SQL-style schemas + read/write ops), tasks, and rubrics.
  - **User-oriented simulation:** decouple *Task* from *User* by adding a dedicated user simulator with behavioral rules (e.g., one subtask at a time; turn-by-turn feedback), forcing longer interactions.
  - **Plug-and-play initialization:** can start generation “from any state” (e.g., inject tool requirements into an ongoing conversation), enabling scalable augmentation.
- **High-density trajectories:** allow multiple task completions in one conversation thread (closer to real sessions).

## What are the key metrics?

- Multi-turn tool-use performance on agentic benchmarks (named in the paper):
  - BFCL
  - ττ2 (tau-tau2)
- “Consistency analysis under repeated executions” (reported qualitatively in the intro): whether correct tool-use persists across multiple trials vs isolated successes.

## What are the main results?

- Models trained on their generated data show **stronger multi-turn performance** and **more reliable tool usage**, especially for **long-horizon** and **stateful** domains.
- Repeated-run analysis suggests improved **consistency** (sustained correct behavior across trials).

## How is this similar to GALILEO?

- Shares the core concern that **multi-turn interaction quality** matters and that single-turn/short-horizon evaluation can be misleading.
- Emphasizes **trajectory-level** properties (long-horizon, consistency across runs), which aligns with GALILEO’s focus on dynamics rather than one-off answers.

## How is this different from GALILEO?

- This is primarily about **data generation for tool-use agents**, not about **pressure-driven drift/sycophancy** or disentangling evidence-based revision vs social influence.
- Evaluation is benchmark-driven (tool-use success), rather than explicitly modeling *time-to-failure*, flips, recovery, or operator families.

## Where GALILEO is stronger / cleaner (if true)

- GALILEO-style protocols can provide **clearer causal stress tests** for *why* behavior changes across turns (pressure vs evidence vs interventions), beyond “does the agent complete the task.”

## Where GALILEO is weaker / needs to improve

- If GALILEO includes agent/tool-use settings, we may need stronger **realism knobs** (user feedback loops, incremental requests) to avoid over-optimistic “efficient solver” trajectories.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider a **user-simulator layer** for any GALILEO tool-use or long-horizon settings: enforce incremental subgoals + feedback to elicit realistic turn counts.
- [ ] Add a small “**repeat-run consistency**” check (same initial state, multiple stochastic runs) to quantify whether robustness is stable or luck-based.

## Quotes / details to potentially cite

- Motivation: task-oriented simulation can yield “**solely task-solving** trajectories” with minimal interaction; propose “**user-oriented simulation**” with incremental requests + turn-by-turn feedback to produce extended dialogues.
