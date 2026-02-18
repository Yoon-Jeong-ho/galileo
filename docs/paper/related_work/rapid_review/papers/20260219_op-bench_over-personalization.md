# OP-Bench: Benchmarking Over-Personalization for Memory-Augmented Personalized Conversational Agents

- Year: 2026
- Venue: arXiv
- Authors: Yulin Hu, Zimo Long, Jiahe Guo, Xingyu Sui, Xing Fu, Weixiang Zhao, Yanyan Zhao, Bing Qin
- URL: https://arxiv.org/abs/2601.13722
- BibTeX key (if we add it): hu2026opbench
- Tags: personalization, memory-augmented agents, benchmark, over-personalization, sycophancy, evaluation

## One-sentence takeaway

OP-Bench is a 1.7k-instance benchmark that measures *over-personalization* (irrelevance, repetition, sycophancy) in memory-augmented dialogue agents and introduces a simple relevance-based memory filter (Self-ReCheck) to reduce it.

## What problem does it solve?

- Existing personalization/memory benchmarks mostly test whether agents *can* recall and use user memories, but not whether they use them *appropriately*.
- With long-term memory, agents can become intrusive/forced: injecting irrelevant personal facts, repeating the same memories across different queries, or agreeing with the user inappropriately (sycophancy).

## What is the core method / protocol?

- **Problem formalization:** three over-personalization (OP) categories:
  - **Irrelevance:** unnecessary injection of user memory/profile when the query does not call for personalization.
  - **Repetition:** overly similar responses / repeatedly invoking the same user memories across semantically distinct queries.
  - **Sycophancy:** over-deference to user beliefs/memories/values at the cost of objectivity or factual accuracy.
- **Benchmark construction (OP-Bench):**
  - Built from **LoCoMo** long-horizon multi-session dialogues (20 users).
  - Pipeline: (1) extract user profile/topics, (2) generate task queries targeting the 3 OP modes (with subtypes), (3) human verification (double review + adjudication).
  - Dataset stats: 1,700 total instances; Repetition dominates (~52%).
- **Evaluation:** uses automated (LLM-based) scorers for Irrelevance and Sycophancy; repetition computed from embedding cosine similarity across responses.
- **Mitigation (Self-ReCheck):** lightweight, model-agnostic memory filtering that selects/keeps memories based on relevance to the current query (plug-and-play in front of the memory-augmented agent).

## What are the key metrics?

- **Irrelevance score:** LLM-judge score in [0, 1] (higher is better; less irrelevant personalization).
- **Sycophancy score:** LLM-judge score in [0, 1] (higher is better; more resistant).
- **Repetition score:** 1 - mean pairwise cosine similarity of response embeddings (higher is better; more diverse / less repetitive).
- Paper also discusses analysis signals such as memory attention / attribution (e.g., “memory-to-query attention ratio”).

## What are the main results?

- Adding memory mechanisms **consistently worsens** OP scores vs a BASE (no-memory) agent across multiple models and memory systems.
- Over-personalization is reported as **widespread**: relative drops vs BASE roughly **26% to 61%** depending on model/system configuration.
- **Self-ReCheck** reportedly reduces over-personalization by **~29%** while largely preserving personalization performance.

## How is this similar to GALILEO?

- If GALILEO includes personalization, user modeling, long-term memory, or retrieval of user context, OP-Bench directly targets a key failure mode: “memory hijacking” (retrieved personal facts dominating generation).
- The taxonomy (irrelevance/repetition/sycophancy) offers a concrete structure for analyzing *when* memory use becomes harmful.

## How is this different from GALILEO?

- OP-Bench is primarily an **evaluation benchmark + diagnostic framing** for personalized dialogue agents, not a new end-to-end agent architecture.
- The mitigation (Self-ReCheck) is positioned as **lightweight filtering**, not a full redesign of memory writing/reading, preference learning, or planning.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO already uses explicit relevance gating / query-intent classification / memory salience constraints, it may be less prone to the “always retrieve and stuff memories” pattern that OP-Bench diagnoses.

## Where GALILEO is weaker / needs to improve

- If GALILEO’s memory retrieval is optimized mainly for “use memory often” (or recall metrics), it may inadvertently increase irrelevance/repetition/sycophancy.
- If sycophancy is handled only via generic safety prompts, persistent memory could still amplify over-alignment (memory-level sycophancy in particular).

## Action items for GALILEO (experiments / method / writing)

- [ ] Add an “over-personalization” subsection to related work + motivation: cite OP-Bench taxonomy (irrelevance/repetition/sycophancy) as a missing axis in many memory benchmarks.
- [ ] Evaluate GALILEO on OP-Bench-style probes (even if not the exact dataset):
  - Irrelevance: random non-personal queries where memory should be unused.
  - Repetition: clusters of related-but-distinct queries; check response diversity.
  - Sycophancy: fact/value probes where user profile could bias agreement.
- [ ] Implement a simple **memory filter / re-check** stage (Self-ReCheck-like): retrieve K, then re-rank/filter by query relevance; optionally add a “none of the above” decision.
- [ ] Track a “memory-to-query attention / attribution” proxy (or simpler: measure how often memory is cited/mentioned when it shouldn’t be).

## Quotes / details to potentially cite

- “We formalize over-personalization into three types: **Irrelevance, Repetition, and Sycophancy** … and introduce **OP-Bench** … of **1,700** verified instances … from long-horizon dialogue histories.”
- “Across models and memory systems, [Self-ReCheck] reduces over-personalization by **29%** while largely preserving personalization … [and] lowers … the memory-to-query attention ratio.”
