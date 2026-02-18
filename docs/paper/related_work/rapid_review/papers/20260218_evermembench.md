# EverMemBench: Benchmarking Long-Term Interactive Memory in Large Language Models

- Year: 2026
- Venue: arXiv
- Authors: Chuanrui Hu; Tong Li; Xingze Gao; Hongda Chen; Yi Bai; Dannong Xu; Tianwei Lin; Xinda Zhao; Xiaohong Li; Yunyun Han; Jian Pei; Yafeng Deng
- URL: https://arxiv.org/abs/2602.01313
- BibTeX key (if we add it): Hu2026EverMemBench
- Tags: memory, long-term, interactive, multi-party, benchmark

## One-sentence takeaway

EverMemBench is a large-scale (1M+ tokens) multi-party conversational memory benchmark that exposes sharp failures in multi-hop, temporal, and retrieval-driven “memory awareness” evaluation for LLM assistants.

## What problem does it solve?

- Existing long-term memory benchmarks for assistants are mostly dyadic and single-topic, and don’t stress realistic conditions like multi-party/group dynamics, topic interleaving, evolving facts, and role/persona constraints.
- The field lacks a standardized, difficult testbed to compare memory systems on *what to remember*, *when it changed*, and *whether the system knows it needs memory*.

## What is the core method / protocol?

- Introduces **EverMemBench**: multi-party, multi-group conversations spanning **>1 million tokens**, with temporally evolving information and cross-topic interleaving.
- Provides **1,000+ QA pairs** and evaluates memory systems along three dimensions:
  - **Fine-grained recall** (retrieve/answer specific details)
  - **Memory awareness** (identify that relevant memory exists; retrieval bottleneck)
  - **User profile understanding** (persona/role-specific information)

## What are the key metrics?

- QA accuracy across the three evaluation dimensions (recall / awareness / profile).
- Emphasis on stressors:
  - **Multi-hop reasoning** in multi-party settings
  - **Temporal reasoning** with “version semantics” (facts change)
  - **Retrieval quality** under semantic gap between query and implicitly-relevant memories

## What are the main results?

- Multi-hop reasoning “collapses” in multi-party settings; even an **oracle** setting reportedly achieves only **~26%**.
- Temporal reasoning remains unsolved; simple timestamp-based approaches are insufficient and require reasoning about *versions* of facts.
- Memory awareness is bottlenecked by retrieval: similarity-based retrieval often fails when relevance is implicit.

## How is this similar to GALILEO?

- Same broad target: **LLM assistants with long-horizon memory**.
- Highlights evaluation dimensions GALILEO likely cares about: long-context interactions, changing user/world state, and retrieval+reasoning coupling.

## How is this different from GALILEO?

- Primarily an **evaluation benchmark** (multi-party, 1M-token scale), rather than a proposed assistant/memory architecture.
- Focuses explicitly on **multi-party/multi-group** conversational settings and role/persona constraints.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO offers clearer architectural interventions (e.g., structured memory, better retrieval/routing, explicit temporal/version modeling), it can position itself as addressing exactly the failure modes EverMemBench surfaces.

## Where GALILEO is weaker / needs to improve

- If GALILEO evaluation is mostly dyadic or short-horizon, EverMemBench suggests it may not reflect real-world complexity.
- If GALILEO relies heavily on vanilla similarity retrieval, EverMemBench indicates this may be the main bottleneck for “memory awareness”.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a related-work paragraph framing EverMemBench as evidence that **multi-party + long-horizon + evolving facts** are not handled by current memory systems.
- [ ] Consider adding (or emulating) an evaluation slice for **memory awareness**: can the system retrieve implicitly-relevant memories that are not lexically similar to the query?
- [ ] If GALILEO has temporal components, emphasize **version semantics** (facts that update) rather than timestamp heuristics.

## Quotes / details to potentially cite

- “multi-party, multi-group conversations spanning over 1 million tokens”
- Evaluates three dimensions: “fine-grained recall, memory awareness, and user profile understanding”
- Key findings: multi-hop collapse (oracle ~26%); temporal reasoning needs “version semantics”; retrieval similarity fails to bridge semantic gap for implicit relevance
