# ENGRAM: Effective, Lightweight Memory Orchestration for Conversational Agents

- Year: 2025
- Venue: arXiv (cs.MA)
- Authors: Daivik Patel; Shrenik Patel
- URL: https://arxiv.org/abs/2511.12960
- BibTeX key (if we add it): patel2025engram
- Tags: memory, long-horizon, orchestration, dense-retrieval, typing, sqlite

## One-sentence takeaway

ENGRAM shows that a *very simple* typed-memory design (episodic/semantic/procedural) + dense retrieval can match or beat more complex long-term memory orchestration systems on LoCoMo and LongMemEval.

## What problem does it solve?

- LLM chat systems need long-horizon consistency (remember events, preferences, instructions) beyond the context window.
- Many existing "agent memory" stacks are architecturally complex (graphs, multi-stage retrieval, schedulers), which hurts reproducibility/engineering overhead.

## What is the core method / protocol?

- Convert each user turn into one or more **typed memory records**:
  - **Episodic**: event-like information (title, summary, time anchor).
  - **Semantic**: stable facts/preferences (fact string + time anchor).
  - **Procedural**: instructions/workflows (title + normalized content + time anchor).
- A **router** outputs a 3-bit mask indicating which store(s) the turn should update.
- Store records + embeddings in a simple DB (they mention local **SQLite**).
- At query time:
  - Embed query, retrieve **top-k dense neighbors per type** (cosine similarity).
  - Merge/deduplicate with **simple set operations**; feed retrieved snippets as evidence context.

## What are the key metrics?

- QA accuracy / score on long-horizon conversational memory benchmarks:
  - **LoCoMo** (multi-session conversational QA).
  - **LongMemEval** (extended-horizon conversational benchmark).
- Efficiency proxy: tokens used vs full-context baseline.

## What are the main results?

- Claims state-of-the-art on **LoCoMo**.
- On **LongMemEval**, exceeds full-context baseline by **+15 absolute points** while using **~1%** of the tokens.
- Main qualitative claim: careful typing + straightforward dense retrieval can rival heavier architectures.

## How is this similar to GALILEO?

- If GALILEO is also about long-term conversational memory: shared framing (consistency across sessions; grounding answers in past events/preferences/instructions).
- Similar retrieval-augmented pattern: external memory store → retrieve evidence → inject into answering prompt.

## How is this different from GALILEO?

- ENGRAM explicitly constrains memory into **three canonical types** with **normalized schemas** and a **single router + single retriever**; it argues against elaborate multi-component orchestration.
- Implementation emphasis: lightweight / reproducible (SQLite + dense retrieval + simple merging) rather than graph traversal or OS-style scheduling.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO provides richer structure (e.g., explicit relational links, temporal reasoning, lifecycle policies), it may outperform ENGRAM on tasks requiring structured multi-hop reasoning beyond similarity search.

## Where GALILEO is weaker / needs to improve

- Ensure GALILEO’s design doesn’t over-complicate the baseline: ENGRAM is a strong argument that *typed dense retrieval* is a necessary comparison point.
- Token-efficiency reporting: ENGRAM highlights large gains over full-context baselines at tiny token budgets.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add/strengthen a **typed-memory baseline** (episodic/semantic/procedural) with a minimal router and dense retrieval; use as a "simplicity" comparator.
- [ ] Report **token budget** and compare to **full-context** baselines; include plots of performance vs context/tokens.
- [ ] If using more complex orchestration, isolate ablations to justify each moving part vs ENGRAM-like minimalism.

## Quotes / details to potentially cite

- "We present ENGRAM, a lightweight state-of-the-art memory system that organizes conversation into three canonical memory types—episodic, semantic, and procedural—through a single router and retriever." (abstract)
- "ENGRAM attains state-of-the-art results on the LoCoMo benchmark … and exceeds the full-context baseline by 15 absolute points on LongMemEval … while using only ≈1% of the tokens." (abstract)
- Typed record sketches (as presented): episodic (title, summary, time anchor, embedding), semantic (fact, time anchor, embedding), procedural (title, normalized content, time anchor, embedding). (Sec. 3.3)
