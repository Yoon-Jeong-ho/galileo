# Agent World Model: Infinity Synthetic Environments for Agentic Reinforcement Learning

- Year: 2026
- Venue: arXiv (paper targets ICML; not sure of acceptance)
- Authors: Canwen Xu; Boyi Liu; Yite Wang; Siwei Han; Zhewei Yao; Huaxiu Yao; Yuxiong He
- URL: https://arxiv.org/abs/2602.10090
- BibTeX key (if we add it): xu2026agentworldmodel
- Tags: agents, synthetic-data, environments, tool-use, multi-turn

## One-sentence takeaway

AWM is an open-source pipeline that uses LLMs + execution/self-correction to synthesize **1,000 code-driven, database-backed MCP tool environments**, enabling scalable multi-turn agent RL with robust verification-style rewards and promising OOD generalization.

## What problem does it solve?

- RL for tool-using agents needs **many diverse interactive environments**; real services are expensive/unstable, and LLM-simulated environments are costly and can hallucinate state transitions.
- Existing “programmed” environment sets are typically **small** and/or rely on human priors (task sets, API docs).

## What is the core method / protocol?

- Environment synthesis decomposed into software-like components:
  - **Scenario** generation (seeded by ~100 domain names; filtered for CRUD/stateful apps; deduplicated).
  - **Task** generation (k=10 tasks per scenario; “API-solvable”; assumes post-authentication).
  - **Database** synthesis: SQLite schema + synthetic seed data such that tasks are executable from initial state.
  - **Tool interface** synthesis: generate an MCP server exposing a Python toolset; tool calls implement DB transitions.
  - **Verification** synthesis: per-task module inspects DB state pre/post; final decision via **code-augmented LLM-as-a-judge** (labels: Completed / Partially Completed / Agent Error / Environment Error).
- Uses **execution-based self-correction** loops: run generated artifacts, feed errors back to the LLM, retry (reported up to 5 iterations).
- Agent RL:
  - Trains multi-turn tool-use agents with **GRPO**.
  - Reward: step-level tool-call format checks (invalid → terminate with -1), plus outcome reward from verification (Completed=1.0, Partial=0.1, else 0.0).
  - Adds **history-truncation during training** to match inference-time history management.

## What are the key metrics?

- Benchmark performance: success / accuracy depending on benchmark category.
- Reward outcomes (Completed vs Partial vs failure) mediated by verification.
- Synthesis pipeline stats: success rate and average self-correction iterations.

## What are the main results?

- Scale: **1,000 environments**, **~10,000 tasks**, **~35k tools** (reported 35,062 tools) with DB-backed state and MCP tool interfaces.
- Synthesis robustness: pipeline reportedly achieves **>85% success rate**, with **~1.13 self-correction iterations** on average.
- OOD evaluation (trained on AWM rather than benchmark-specific environments) on 3 benchmarks:
  - **Verified τ²-bench** (airline/retail/telecom multi-turn tasks)
  - **BFCLv3** (function calling; includes multi-turn + hallucination tests)
  - **MCP-Universe** (real-world MCP servers; multi-server workflows)
- Claims: training exclusively in AWM synthetic environments yields **strong out-of-distribution generalization** vs baselines (e.g., LLM-simulated transitions; smaller synthesized env sets).

## How is this similar to GALILEO?

- Both care about **multi-turn agent behavior** and evaluation that isn’t purely static/offline.
- Similar emphasis on **robust, automatable evaluation signals** (they combine code-based checks + LLM judging).

## How is this different from GALILEO?

- AWM is primarily about **environment synthesis + agent RL training** for tool-use; not about modeling belief drift/persuasion/sycophancy dynamics per se.
- Their “ground truth” is largely **database state transitions**, not human-labeled conversational correctness.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO’s core focus is social pressure / belief dynamics, it likely has cleaner **controls for drift vs evidence-based revision** than AWM’s tool-task framing.

## Where GALILEO is weaker / needs to improve

- If GALILEO requires scalable interactive setups, AWM highlights how far you can get with **code-driven, DB-backed synthetic environments** + verification to enable *cheap iteration* and *repeatable online interaction*.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adopting the **“DB state diff + verifier”** pattern for any GALILEO interactive tasks (even outside tool-use), to reduce evaluator brittleness.
- [ ] If GALILEO trains agents/policies, cite AWM as evidence that **training on synthetic-but-executable environments** can generalize OOD.
- [ ] Use their decomposition (scenario → tasks → state schema → interface → verification) as a writing blueprint for any GALILEO environment/protocol generation section.

## Quotes / details to potentially cite

- Scale/complexity: “With 1,000 environments, 35,062 tools and 10,000 tasks … AWM is the largest open-source tool-use environment set to date.” (arXiv HTML v2, Sec. 2–3)
- Reward design: Completed=1.0, Partially Completed=0.1, else 0.0; invalid tool format → early termination with -1.0 (Sec. 4.1).
- Benchmarks used for OOD generalization: verified τ²-bench, BFCLv3, MCP-Universe (Sec. 5.1).
