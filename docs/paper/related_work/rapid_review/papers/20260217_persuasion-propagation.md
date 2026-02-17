# Persuasion Propagation in LLM Agents

- Year: 2026
- Venue: arXiv (cs.AI, cs.MA)
- Authors: Hyejun Jeong; Amir Houmansadr; Shlomo Zilberstein; Eugene Bagdasarian
- URL: https://arxiv.org/abs/2602.00851
- BibTeX key (if we add it): Jeong2026PersuasionPropagation
- Tags: persuasion, multi-turn, agents, trace-level evaluation, belief-state conditioning, web-research agent, coding agent

## One-sentence takeaway

Task-irrelevant persuasion can measurably change *tool-using agent behavior traces* (search breadth, sources visited, etc.) even when final outputs look fine—especially when a belief/stance is made explicit at task time.

## What problem does it solve?

- Moves “persuasion/sycophancy” evaluation from *stance compliance* to **downstream agent behavior drift**, i.e., whether a persuaded belief state propagates into later, unrelated tasks.
- Highlights an auditing gap: outcome-only scoring can miss subtle but systematic changes in exploration / evidence collection.

## What is the core method / protocol?

- Defines **persuasion propagation**: a belief/stance influenced by persuasion persists and affects later task execution even when irrelevant.
- Proposes a behavior-centered framework that separates:
  - **On-the-fly persuasion**: inject persuasion during/near task execution (competes with task context).
  - **Belief-prefill conditioning**: explicitly set the agent’s belief/stance at task time (belief / disbelief / neutral) to isolate belief effects from timing.
- Evaluates on tool-using agent tasks:
  - **Web research** tasks (search + source browsing).
  - **Coding** tasks (iteration/revision dynamics).

## What are the key metrics?

- Trace / process metrics rather than just final answer quality, e.g.:
  - Number of searches.
  - Number of unique sources visited.
  - (Conceptually) revisions / strategy changes during execution.

## What are the main results?

- **On-the-fly persuasion**: weak and inconsistent downstream behavioral effects.
- **Belief-prefilled at task time**: clear behavioral shifts vs neutral conditioning:
  - ~**26.9% fewer searches** on average.
  - ~**16.9% fewer unique sources** visited.
- Takeaway: persuasion can matter most when the *belief state is made salient at task time*, and effects can show up as **reduced exploration / narrower evidence collection** rather than overtly wrong final outputs.

## How is this similar to GALILEO?

- Same high-level concern: **multi-turn / long-horizon robustness** where prior turns (or prior sessions) influence later behavior.
- Supports GALILEO’s emphasis that robustness should be assessed over **trajectories**, not just final-turn correctness.

## How is this different from GALILEO?

- Focuses on **agent execution behavior** (search breadth, visited sources, tool-use traces) rather than primarily on belief accuracy / flip dynamics.
- Does not (from the abstract/intro) deeply separate **evidence-driven belief revision** vs **pressure-driven drift** using explicit control conditions; it’s more “belief state affects behavior” than “revision vs drift diagnosis.”

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO has explicit **drift-vs-revision controls** and **recovery-after-flip** trajectories, we can claim a cleaner causal decomposition than “behavior changes under conditioning.”
- If GALILEO already reports ToF/survival/flip-quality metrics, it offers more direct *stability* characterization than “fewer searches/sources.”

## Where GALILEO is weaker / needs to improve

- GALILEO should consider adding **agentic trace-level metrics** (search diversity, source diversity, exploration depth) because this paper shows persuasive effects may hide in process even when outputs look acceptable.
- If GALILEO’s evaluations are outcome-only or QA-only, we may miss “silent narrowing” failure modes in realistic tool-using agents.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add an **agent-trace slice**: evaluate whether social pressure/persuasion changes (i) number of searches, (ii) unique domains/sources, (iii) evidence diversity, even when answer quality is unchanged.
- [ ] Mirror their disentangling idea: run both **on-the-fly** vs **belief-salient-at-task-time** conditions (or an explicit “state injection” control) to separate timing/recency artifacts.
- [ ] In writing, motivate why GALILEO evaluates beyond final answers: cite this as evidence that *process drift* is a real, measurable risk.

## Quotes / details to potentially cite

- Defines “persuasion propagation”: task-irrelevant persuasion influencing downstream agent behavior.
- Key quantitative headline (belief-prefill vs neutral): **26.9% fewer searches** and **16.9% fewer unique sources**.
