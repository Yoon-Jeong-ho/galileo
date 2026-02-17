# Unsafer in Many Turns: Benchmarking and Defending Multi-Turn Safety Risks in Tool-Using Agents

- Year: 2026
- Venue: ICML (per arXiv HTML)
- Authors: Xu Li et al. (see arXiv)
- URL: https://arxiv.org/abs/2602.13379
- BibTeX key (if we add it): unsafer_many_turns_2026
- Tags: multi-turn, agent-safety, tool-use, jailbreak, benchmark, defense

## One-sentence takeaway

MT-AgentRisk shows that distributing harmful tool-use intent across multiple benign-looking turns measurably increases attack success, and proposes ToolShield, a training-free “self-exploration” defense that reduces multi-turn tool-use ASR.

## What problem does it solve?

- Existing safety evaluations often cover either (a) multi-turn dialogue jailbreaks without tool execution, or (b) single-turn tool-using harmful tasks; they miss the combined risk where an agent is coaxed over multiple turns into executing harmful tool operations.
- Need a scalable way to transform single-turn harmful tool tasks into realistic multi-turn attack sequences.

## What is the core method / protocol?

- **Multi-turn attack taxonomy** that transforms a single-turn harmful task into a multi-turn sequence along three dimensions:
  - **Format**: Addition vs Decomposition (either add abstraction layers or fragment the harmful task into subtasks).
  - **Method**: two methods per format (paper describes 4 methods total).
  - **Target**: manipulate **Data Files** vs **Environment States**.
  - => 8 subcategories (format × method × target).
- **MT-AgentRisk benchmark**:
  - Curates **365** single-turn harmful tasks (from prior benchmarks) and converts them into multi-turn attack sequences using the taxonomy.
  - Covers five tools: **Filesystem**, **Browser** (Playwright), **PostgreSQL**, **Notion**, **Terminal**.
- **ToolShield defense** (training-free): when a new tool is introduced, the agent
  1) reads/analyzes tool documentation to propose test cases,
  2) executes test cases in a sandbox to observe downstream effects,
  3) distills “safety experiences” for use during deployment (tool-agnostic guidance).
  - Also reports cross-model transfer of safety experiences.

## What are the key metrics?

- **Attack Success Rate (ASR)**: fraction of tasks where the agent executes the harmful outcome (or completes the harmful tool-use goal) under the attack sequence.
- Delta ASR between single-turn vs multi-turn settings (safety degradation due to multi-turn decomposition/addition).

## What are the main results?

- **Multi-turn makes agents less safe**: ASR increases by ~**16% on average** across evaluated open + closed models in multi-turn settings (paper also lists larger increases for some models).
- **ToolShield helps**: reduces ASR by ~**30% on average** in multi-turn interactions (reports larger reductions for some models).

## How is this similar to GALILEO?

- Shares the core concern that **multi-turn interaction dynamics** reveal failures hidden by single-turn evaluations.
- Reinforces the high-level narrative: safety/robustness degrades over turns as adversaries can “shape” the trajectory.

## How is this different from GALILEO?

- Focuses on **tool-using agent safety** (harmful actions via tools), not belief drift / sycophancy / persuasion-driven flip dynamics.
- Primary endpoint is **task success at causing harm (ASR)** rather than time-to-flip / recovery / pressure-vs-evidence controls.

## Where GALILEO is stronger / cleaner (if true)

- GALILEO’s framing around **pressure-only drift vs evidence-driven revision** and **recovery dynamics** is orthogonal and can provide finer-grained behavioral diagnostics than a binary ASR endpoint.

## Where GALILEO is weaker / needs to improve

- If GALILEO wants to make claims about “realistic multi-turn risk,” this paper highlights a key adjacent axis: **tool execution** and **stateful environments**.

## Action items for GALILEO (experiments / method / writing)

- [ ] Related-work positioning: cite MT-AgentRisk as evidence that **multi-turn decomposition** systematically increases attack success, motivating long-horizon protocols.
- [ ] Consider a small appendix note: how GALILEO-style drift metrics might extend to **tool-using agents** (e.g., time-to-unsafe-action / survival-to-policy-violation).
- [ ] Borrow “taxonomy for transforming single-turn → multi-turn” as a general recipe (even if applied to belief/pressure tasks rather than tool harms).

## Quotes / details to potentially cite

- Proposes **MT-AgentRisk**, “the first benchmark to evaluate multi-turn tool-using agent safety.”
- Reports **ASR increases by ~16% on average** in multi-turn settings.
- Proposes **ToolShield**, a training-free self-exploration defense that reduces ASR by ~30% on average in multi-turn interactions.
