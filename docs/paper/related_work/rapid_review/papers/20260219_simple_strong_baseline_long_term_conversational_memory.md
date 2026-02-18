# A Simple Yet Strong Baseline for Long-Term Conversational Memory of LLM Agents

- Year: 2025
- Venue: arXiv (work in progress; labeled “Machine Learning, ICML” on arXiv HTML)
- Authors: Sizhe Zhou et al. (not fully listed on arXiv page extract)
- URL: https://arxiv.org/abs/2511.17208
- BibTeX key (if we add it): zhou2025emem (placeholder)
- Tags: memory, long-term, multi-session, personalization, robustness

## One-sentence takeaway

Represent multi-session dialogue memory as event-like “enriched EDUs” (self-contained propositions) and retrieve them (optionally with a lightweight graph+PPR step) to outperform strong long-term memory baselines with shorter QA contexts.

## What problem does it solve?

- Long-term conversational agents need to recall fine-grained details across many sessions, but:
  - prompt context windows are finite,
  - coarse chunk retrieval misses details,
  - turn-level retrieval fragments context,
  - summarization/fact distillation can be lossy and drop “minor” details that later matter.

## What is the core method / protocol?

- Offline “write”/indexing:
  - Use an LLM to decompose each session into **enriched elementary discourse units (EDUs)**: short event-like propositions that may fuse info across turns, normalize entities, attach timestamps (possibly inferred from session time), and include source-turn attributions.
  - (For a graph variant) For each EDU, extract event arguments (free-form role–argument pairs; no fixed ontology).
  - Build a heterogeneous graph with nodes: sessions, EDUs, arguments; edges: session–EDU, EDU–argument, argument–argument “synonymy” edges based on embedding similarity.

- Online “read”/retrieval:
  - Retrieve candidate EDUs by dense similarity to the query embedding.
  - Detect entity/concept mentions in the query; retrieve candidate argument nodes per mention by similarity.
  - Apply a **recall-oriented LLM filter** over candidate EDUs and arguments (keep borderline items; rely on later steps to down-weight).
  - Two variants:
    - **EMem**: dense EDU retrieval + recall-oriented LLM filtering (no graph).
    - **EMem-G**: same retrieval/filtering + **Personalized PageRank** propagation from seed EDUs/arguments to select a small, graph-consistent set of EDUs for the QA context.

## What are the key metrics?

- Benchmark task performance on long-term conversational memory QA:
  - LoCoMo
  - LongMemEval_S
- Efficiency proxy: ability to answer with **shorter QA contexts** (retrieve fewer/smaller memory items).

## What are the main results?

- On **LoCoMo** and **LongMemEval_S**, event-centric EDU memories (EMem / EMem-G) **match or surpass strong baselines**.
- Achieves competitive/better results while using **much shorter QA contexts** (compared to baselines that need longer retrieved contexts).

## How is this similar to GALILEO?

- Emphasizes structured long-term memory for agents (beyond raw transcript concatenation).
- Uses a two-stage pipeline: write-side structuring + read-side retrieval/ranking.
- Optional graph-based associative recall (multi-hop evidence aggregation).

## How is this different from GALILEO?

- Primary memory unit is an **event-like natural-language proposition (enriched EDU)**, explicitly motivated by neo-Davidsonian event semantics, rather than (only) summaries, turn chunks, or strict KG triples.
- Uses **LLM filtering tuned for recall** as an explicit step (vs. purely similarity thresholds or precision-heavy filters).
- Graph is heterogeneous (session/EDU/argument) with synonymy edges over arguments; retrieval seeds include both EDUs and arguments.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO uses stronger provenance/grounding guarantees than “mildly abstractive” EDUs, it may be cleaner for faithfulness and auditability.
- If GALILEO avoids LLM-based indexing steps (EDU/argument extraction), it may be simpler/cheaper/less brittle.

## Where GALILEO is weaker / needs to improve

- If GALILEO relies on coarse chunking or heavy summarization, this paper suggests a middle ground: **non-compressive, fine-grained, self-contained memory atoms** that preserve more detail.
- If GALILEO lacks associative graph propagation, EMem-G’s PPR step is a lightweight add-on to improve multi-hop recall.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider an “event-proposition memory” baseline: rewrite sessions into self-contained event-like statements with normalized entities + timestamps + provenance.
- [ ] Try a recall-oriented filter stage (LLM or heuristic) explicitly biased to keep borderline candidates before a final scorer.
- [ ] If a graph memory exists: test PPR from seed nodes (EDUs + argument concepts) to improve multi-session associative recall.
- [ ] In related work: position against triple-based KGs and lossy summarization approaches; cite the EDU-as-event unit as a principled middle granularity.

## Quotes / details to potentially cite

- “Motivated by neo-Davidsonian event semantics, we propose an event-centric alternative that represents conversational history as short, event-like propositions…”
- They “avoid lossy compression” and instead preserve information in a non-compressive form via enriched EDUs with normalized entities and source turn attributions.
- Two variants: EMem (dense EDU retrieval + LLM filter) and EMem-G (adds graph + Personalized PageRank propagation).
