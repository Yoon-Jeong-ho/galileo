# MEM1: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents

- Year: 2025
- Venue: arXiv
- Authors: Zijian Zhou*, Ao Qu*, Zhaoxuan Wu, Sunghwan Kim, Alok Prakash, Daniela Rus, Jinhua Zhao, Bryan Kian Hsiang Low, Paul Pu Liang
- URL: https://arxiv.org/abs/2506.15841
- BibTeX key (if we add it): mem1_zhou_2025
- Tags: agents, memory, reasoning, long-horizon, reinforcement-learning

## One-sentence takeaway

MEM1 trains an LLM agent (via end-to-end RL) to maintain a compact *internal state* that merges reasoning + memory consolidation so multi-turn tool-using tasks can run with ~constant prompt/memory rather than ever-growing full-history contexts.

## What problem does it solve?

- Long-horizon interactive agents commonly append all past turns (thoughts/actions/observations) into the prompt, causing:
  - increasing inference cost / KV cache memory,
  - degradation when context lengths exceed training distribution,
  - attention dilution from irrelevant/redundant history.
- Prior “external memory module” approaches often require separate retriever/summarizer components that are not optimized end-to-end with the agent policy.

## What is the core method / protocol?

- Introduce an explicit *internal state* token block (paper describes it as enclosed in something like `<IS> ... </IS>`) that is the **only retained** memory between turns.
- At each turn, the agent:
  1) reads the current internal state + current user query (and any fixed system prompt),
  2) optionally uses tools / observes environment outputs,
  3) writes an updated internal state that consolidates relevant info and discards the rest,
  4) discards prior turn context/tool outputs (preventing prompt growth).
- Train this behavior end-to-end with reinforcement learning using verifiable rewards for task success (memory efficiency is largely an emergent behavior from the policy structure rather than an explicit reward term, per intro).
- Environment construction: propose a scalable *multi-turn task augmentation* approach that composes existing single-objective datasets into longer sequences (multi-objective / multi-hop across turns) to train/evaluate long-horizon behavior.

## What are the key metrics?

- Task success / accuracy on multi-turn tasks (including multi-hop QA objectives).
- Memory usage / context length growth proxy (they report “reducing memory usage by 3.7x” vs baseline in at least one setting).
- Generalization beyond training horizon (performance as number of turns/objectives increases past training).

## What are the main results?

- Across three domains mentioned in the abstract:
  - internal retrieval QA,
  - open-domain web QA,
  - multi-turn web shopping,
  MEM1 shows improved long-horizon performance with constant-memory operation.
- Highlighted comparison (abstract): MEM1-7B improves performance by **3.5x** while reducing memory usage by **3.7x** compared to **Qwen2.5-14B-Instruct** on a **16-objective multi-hop QA** task.
- Claims improved generalization beyond the training horizon (handles longer sequences than seen during training).

## How is this similar to GALILEO?

- Same problem framing: long-horizon agents need principled memory management rather than “stuff everything into the prompt”.
- Emphasizes that memory is not just storage but interacts with reasoning (working memory / consolidation).

## How is this different from GALILEO?

- MEM1’s mechanism is primarily **policy-learning + an internal-state interface** (single-model end-to-end RL) rather than modular memory components.
- The paper’s “internal state” is a learned latent textual state (still in-token space) that is updated every step; if GALILEO separates memory representations, indexing, or retrieval explicitly, MEM1 is more “monolithic”.
- MEM1 leans heavily on RL + constructed multi-turn compositions; if GALILEO targets more controlled memory semantics (schemas, provenance, citations, edit operations), MEM1 may be lighter on explicit structure.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO provides explicit memory objects, provenance, and/or retrieval criteria, it may be more interpretable/auditable than a single evolving internal-state blob.
- If GALILEO avoids RL reliance (or reduces RL engineering burden), it may be easier to reproduce and iterate.

## Where GALILEO is weaker / needs to improve

- GALILEO should clearly address the “constant-memory across turns” baseline and articulate what is retained vs discarded per step (MEM1 provides a crisp story and interface).
- If GALILEO currently relies on long-context prompting, MEM1 is a strong related-work contrast that motivates moving away from full-history prompts.

## Action items for GALILEO (experiments / method / writing)

- [ ] In related work, contrast “external memory module” vs “end-to-end learned consolidation” and position GALILEO accordingly.
- [ ] Add an experiment axis for *horizon length generalization* (evaluate beyond training or tuning horizon) and explicitly report context/memory growth.
- [ ] Consider adopting MEM1-like evaluation: compose existing single-turn benchmarks into multi-objective sequences to stress long-horizon memory.

## Quotes / details to potentially cite

- Abstract framing: “most LLM systems rely on full-context prompting… unbounded memory growth… degraded reasoning performance on out-of-distribution input lengths.”
- Key claim: “end-to-end reinforcement learning framework that enables agents to operate with constant memory across long multi-turn tasks.”
- Highlighted number: “MEM1-7B improves performance by 3.5× while reducing memory usage by 3.7× compared to Qwen2.5-14B-Instruct on a 16-objective multi-hop QA task.”
