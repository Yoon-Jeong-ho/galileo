# MemoryArena: Benchmarking Agent Memory in Interdependent Multi-Session Agentic Tasks

- Year: 2026
- Venue: arXiv
- Authors: Zexue He, Yu Wang, Churan Zhi, Yuanzhe Hu, Tzu-Ping Chen, Lang Yin, Ze Chen, Tong Arthur Wu, Siru Ouyang, Zihan Wang, Jiaxin Pei, Julian McAuley, Yejin Choi, Alex Pentland
- URL: https://arxiv.org/abs/2602.16313
- BibTeX key (if we add it): He2026MemoryArena
- Tags: agents, memory, multi-session, long-horizon, benchmark

## One-sentence takeaway

MemoryArena proposes a “memory-agent-environment loop” benchmark where agents must **distill experience into memory in earlier sessions** and **use it to act correctly in later sessions**, exposing a gap between long-context recall benchmarks and agentic, decision-coupled memory use.

## What problem does it solve?

- Existing “agent memory” evaluations often split into:
  - recall-only tests (did the model remember a fact from prior text/conversation?), and
  - single-session agent tasks (can the agent act, without requiring long-term memory).
- In realistic deployments, memory is *acquired through interaction* and then *used to drive later actions*; the paper targets this coupled setting.

## What is the core method / protocol?

- Introduces **MemoryArena**, an evaluation “gym” for **multi-session** agent tasks with **explicitly interdependent subtasks**.
- Protocol (as described in the abstract):
  - Session(s) 1..k: agent acts, receives feedback, and must **distill** key experiences into a memory store.
  - Later session(s): agent must **retrieve/use** that stored memory to make correct decisions/actions and complete the overall task.
- Task coverage (abstract):
  - web navigation
  - preference-constrained planning
  - progressive information search
  - sequential formal reasoning

## What are the key metrics?

- The abstract does not specify exact metrics; likely task success / completion accuracy per domain, possibly across sessions (to verify that success requires memory, not just short-context inference).

## What are the main results?

- Agents that are near-saturated on existing long-context memory benchmarks (example cited: **LoCoMo**) can still perform poorly in MemoryArena’s agentic setting.
- Takeaway: current “memory benchmarks” may overestimate readiness for memory-driven agents; MemoryArena identifies a missing evaluation axis (memory *use* for action).

## How is this similar to GALILEO?

- If GALILEO targets robustness/faithfulness over **multi-turn / long-horizon interactions**, MemoryArena is aligned as an evaluation framing where state evolves and earlier interactions influence later decisions.
- Emphasizes **trajectory-level** behavior rather than single-turn correctness.

## How is this different from GALILEO?

- MemoryArena is primarily a **benchmark/evaluation gym** for memory-in-agent loops, not a method for reducing failure modes.
- Its focus is **multi-session memory acquisition and later use**, whereas GALILEO may target other longitudinal dynamics (e.g., drift, pressure, multi-turn attacks, stability) beyond memory.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO provides concrete interventions/controls for longitudinal failures (not just evaluation), it can position itself as “method + analysis,” with MemoryArena as “evaluation target.”

## Where GALILEO is weaker / needs to improve

- If GALILEO currently evaluates mostly via long-context recall or single-session tasks, it may miss the *decision-coupled* multi-session memory loop that MemoryArena claims is crucial.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add MemoryArena to related-work as a benchmark that distinguishes **recall** vs **memory-for-action**.
- [ ] In the paper narrative, explicitly call out the “coupling” critique: *memorization* and *action* should be evaluated jointly.
- [ ] If feasible, include (or at least discuss) a GALILEO-style analysis on a multi-session benchmark setup (even a small-scale proxy) to show relevance beyond static long-context tests.

## Quotes / details to potentially cite

- “Existing evaluations of agents with memory typically assess memorization and action in isolation.”
- MemoryArena: “benchmarking agent memory in multi-session Memory-Agent-Environment loops.”
- It “reveals that agents with near-saturated performance on existing long-context memory benchmarks like LoCoMo perform poorly in our agentic setting.”
