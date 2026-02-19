# TiMem: Temporal-Hierarchical Memory Consolidation for Long-Horizon Conversational Agents

- Year: 2026
- Venue: arXiv
- Authors: Kai Li; Xuanqing Yu; Ziyi Ni; Yi Zeng; Yao Xu; Zheqing Zhang; Xin Li; Jitao Sang; Xiaogang Duan; Xuelei Wang; Chengbao Liu; Jie Tan
- URL: https://arxiv.org/abs/2601.02845
- BibTeX key (if we add it): timem2026
- Tags: agent, memory, long-horizon, consolidation, temporal, hierarchy

## One-sentence takeaway

TiMem proposes a **temporal-first hierarchical memory** (Temporal Memory Tree) with **prompted consolidation** and **query-complexity–aware recall + gating**, showing strong benchmark gains while cutting recalled context.

## What problem does it solve?

- Long-horizon conversational agents accumulate interaction history beyond LLM context windows; naïve semantic clustering / flat retrieval tends to yield **fragmented**, temporally-misaligned memories and unstable personalization.
- Need: (i) temporal coherence over evolving user state, (ii) progressive abstraction from episodes → stable persona, (iii) efficient recall without losing correctness.

## What is the core method / protocol?

- **Temporal Memory Tree (TMT)**: a multi-level tree where each node has a **time interval** and a **memory text/embedding**; parent intervals contain children (explicit temporal containment).
- **5-level hierarchy (implementation choice)**:
  - L1 segment (fine-grained, online)
  - L2 session
  - L3 day
  - L4 week
  - L5 profile/persona (monthly updates)
- **Instruction-guided consolidation (no fine-tuning)**:
  - Each level i uses a level-specific instruction prompt \(\mathcal{I}_i\) to consolidate child memories (plus a small sliding history window, w=3) into a higher-level node.
  - Scheduling: L1 online (after each user-assistant exchange); L2–L5 scheduled at temporal window boundaries.
- **Complexity-aware recall**:
  - LLM **planner** labels query as {simple, hybrid, complex} + extracts keywords.
  - **Hierarchical recall**: activate L1 leaves via fused semantic+lexical scoring (cosine + BM25), then pull ancestors at levels chosen by the planner.
  - **Recall gating**: LLM filters the candidate set to “retain” only memories deemed necessary for the query; final ordering emphasizes hierarchy level + temporal proximity.

## What are the key metrics?

- Primary: LLM-as-a-judge (LLJ) **accuracy** on long-term memory QA benchmarks.
- Efficiency: recalled context length (tokens) and recall latency (P50/P95; reported as seconds).

## What are the main results?

- **LoCoMo**: 75.30% LLJ accuracy; reported **52.20% reduction** in recalled memory length vs baseline comparisons (and ~511 tokens/query in their planned setting).
- **LongMemEval-S**: 76.88% LLJ accuracy (and higher with a stronger answer model).
- Ablations suggest:
  - Planner improves the accuracy–cost trade-off vs fixed-scope recall.
  - Hierarchical propagation matters a lot vs flat recall when only low-level memories exist.
  - Overly narrow recall + aggressive gating can drop accuracy (gating works best when candidate sets include distractors).

## How is this similar to GALILEO?

- If GALILEO needs multi-turn / long-horizon behavior, TiMem is a strong adjacent design point for **external memory**, especially:
  - explicit temporal structuring,
  - progressive consolidation (episode → patterns → persona),
  - adaptive recall budgeting (planner) and redundancy/conflict filtering (gating).

## How is this different from GALILEO?

- TiMem is primarily a **memory architecture + retrieval/consolidation recipe** (and benchmark eval), not a social-pressure / persuasion / sycophancy protocol.
- Relies on multiple LLM calls at recall time (planner + gating), which may be heavy depending on GALILEO’s constraints.
- Uses **LLJ** as the main correctness metric; may not align with GALILEO’s target outcomes if GALILEO focuses on behavioral robustness metrics.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO uses behavioral, multi-turn stress tests (e.g., pressure/recovery), it may offer **cleaner causal attribution** than benchmark LLJ QA.
- If GALILEO avoids additional LLM middleware calls (planner/gating), it may be simpler operationally.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently uses primarily semantic clustering / flat retrieval, TiMem suggests temporal structure should be a **first-class constraint**, not metadata.
- If GALILEO lacks an explicit persona-layer abstraction, TiMem provides a concrete “profile layer” target with periodic updates.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a related-work paragraph contrasting **semantic-first hierarchical retrieval (RAPTOR-style)** vs **temporal-first hierarchical memory (TiMem/TMT)**.
- [ ] Consider a lightweight variant of **temporal containment constraints** (even without a full tree) and evaluate its impact on long-horizon consistency.
- [ ] If GALILEO already has memory, test a **planner-like recall scope** policy (simple vs complex queries) and report context-length vs accuracy trade-offs.
- [ ] Consider “recall-time forgetting” (gating) ablation: broad candidate set + filter vs narrow retrieval, to see which yields better robustness.

## Quotes / details to potentially cite

- “TiMem is characterized by three core properties: (1) temporal–hierarchical organization … (2) semantic-guided consolidation … without fine-tuning; and (3) complexity-aware memory recall …”
- “TiMem achieves state-of-the-art accuracy … 75.30% on LoCoMo and 76.88% on LongMemEval-S … reducing the recalled memory length by 52.20% on LoCoMo.”
