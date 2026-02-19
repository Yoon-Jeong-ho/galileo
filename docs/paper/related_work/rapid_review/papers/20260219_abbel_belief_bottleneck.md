# ABBEL: LLM Agents Acting through Belief Bottlenecks Expressed in Language

- Year: 2025
- Venue: arXiv
- Authors: Aly Lidayan, Jakob Bjorner, Satvik Golechha, Kartik Goyal, Alane Suhr
- URL: https://arxiv.org/abs/2512.20111
- BibTeX key (if we add it): abbel2025lidayan
- Tags: agents, long-horizon, belief-state, summarization, memory, rl

## One-sentence takeaway

ABBEL turns long-horizon agent interaction into a two-call loop (update a natural-language “belief state”, then act from it), and shows RL can train models to keep beliefs accurate/short and even beat full-context baselines.

## What problem does it solve?

- Long-horizon interactive tasks produce interaction histories that quickly exceed practical context windows.
- Naive summarization/bottlenecks can cause error propagation (bad summaries → bad actions → worse summaries).

## What is the core method / protocol?

- Define a belief bottleneck where the agent maintains a natural-language belief state capturing task-relevant unknowns.
- At each step *t*:
  - **Belief update:** produce posterior belief b_t from (instructions, previous belief b_{t-1}, previous action a_{t-1}, new observation o_{t-1}).
  - **Action selection:** choose next action a_t conditioned only on (instructions, current belief b_t).
- Compare to:
  - **Vanilla:** action from full interaction history.
  - **Belief prompting (ablation):** generate beliefs but still keep full history in context for action selection.
- RL post-training to improve bottleneck behavior:
  - “Belief grading” rewards higher-quality beliefs.
  - “Belief length penalty” rewards more compressed beliefs.

## What are the key metrics?

- Task success rate across multiple multi-step environments.
- Memory / context usage proxy: length of belief state vs length of full history over steps.
- (In RL sections) performance vs memory trade-off under length penalties.

## What are the main results?

- Prompted ABBEL can keep memory use ~constant/slow-growing while producing interpretable belief summaries.
- Bottlenecks are prone to **error propagation**: belief update mistakes can degrade downstream performance vs full-context.
- RL can substantially improve ABBEL:
  - In simplified Wordle with Qwen2.5-7B-Instruct + belief grading, ABBEL reportedly exceeds full-context performance by ~20% while keeping beliefs near-constant length.
  - With belief length penalty on a multi-objective QA setup, ABBEL outperforms MEM1 while using less memory (and the bottleneck lets you trade performance for memory more cleanly).
  - On ColBench (collaborative programming), belief grading helps data efficiency and ABBEL can approach full-context performance at ~half the memory.

## How is this similar to GALILEO?

- Same core concern: **long-horizon agenting requires memory management**; summaries/belief states are a compact interface between steps.
- Emphasizes an interpretable intermediate state (belief) that can be inspected, graded, and potentially used for supervision.

## How is this different from GALILEO?

- ABBEL frames memory as an explicit **belief state bottleneck** updated every step; GALILEO may emphasize different internal state representations, retrieval, or tool-augmented memory.
- ABBEL’s training signals are targeted at belief quality/length; GALILEO may not yet have explicit “belief grading” style rewards.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO uses structured state (schemas, typed slots, verifiable facts) it may be less vulnerable than free-form NL belief text to subtle drift/hallucinated memories.

## Where GALILEO is weaker / needs to improve

- If GALILEO relies on generic summarization without explicit bottleneck training/constraints, ABBEL suggests it may suffer the same error propagation issues.
- If GALILEO does not separately model “state update” vs “action selection”, it may entangle reasoning with state, making compression/steering harder (a critique ABBEL makes of MEM1-style designs).

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider a clean two-stage loop in the paper framing: **state update → act from state**, where the “state” is explicitly constrained and evaluated.
- [ ] Add an ablation mirroring ABBEL’s: (a) full context, (b) belief prompting with full context, (c) belief bottleneck.
- [ ] If doing training, consider explicit reward terms for (i) state accuracy/consistency and (ii) state length/compactness.
- [ ] Evaluate/diagnose error propagation: measure how often the maintained state contradicts observed evidence, and correlate with task failure.

## Quotes / details to potentially cite

- “ABBEL replaces long multi-step interaction history by a belief state, i.e., a natural language summary of what has been discovered about task-relevant unknowns.”
- “However, bottleneck approaches are generally prone to error propagation … due to errors in belief updating.”
- ABBEL step protocol: update prior belief with latest observation to posterior belief, then act using only posterior belief.
