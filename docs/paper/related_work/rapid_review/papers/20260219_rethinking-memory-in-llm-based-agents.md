# Rethinking Memory in LLM based Agents: Representations, Operations, and Emerging Topics

- Year: 2025
- Venue: arXiv (survey)
- Authors: Yiming Du; Wenyu Huang; Danna Zheng; Zhaowei Wang; Sebastien Montella; Mirella Lapata; Kam-Fai Wong; Jeff Z. Pan
- URL: https://arxiv.org/abs/2505.00675
- BibTeX key (if we add it): Du2025RethinkingMemoryAgents
- Tags: survey, memory, agents

## One-sentence takeaway

A survey that reframes “memory for LLM agents” around a taxonomy of **memory representations** (parametric vs contextual) and **six atomic operations** (consolidation, updating, indexing, forgetting, retrieval, condensation), then maps these onto emerging research topics.

## What problem does it solve?

- Prior surveys on agent memory often focus on application scenarios (e.g., personalization) rather than the **mechanistic building blocks** that govern memory dynamics.
- Lack of a shared vocabulary for “what it means to do memory” in LLM-agent systems, making it harder to compare methods, benchmarks, and tools.

## What is the core method / protocol?

- Proposes two orthogonal structuring axes:
  - **Representations:**
    - *Parametric memory* (implicitly stored in model weights)
    - *Contextual memory* (explicit external data; structured/unstructured)
  - **Operations:** Consolidation, Updating, Indexing, Forgetting, Retrieval, Condensation
- Uses the representation×operation lens to organize the literature and highlight four “topic clusters”:
  - Long-term memory
  - Long-context memory
  - Parametric modification
  - Multi-source memory
- Curates pointers to datasets/papers/tools (with a companion public repository).

## What are the key metrics?

- As a survey, it does not introduce a single canonical metric; it emphasizes comparing work through the lens of which **operation(s)** are implemented or evaluated (e.g., retrieval accuracy/latency, forgetting behavior, update consistency), and through which **memory topic cluster** the work targets.

## What are the main results?

- Conceptual contribution: a more “systems-like” decomposition of agent memory into reusable primitives (ops), intended to reduce confusion between retrieval, compression, update, and forgetting.
- Practical contribution: a structured map of relevant benchmarks/tools and how they relate to specific memory operations.

## How is this similar to GALILEO?

- If GALILEO studies multi-turn behavioral dynamics (e.g., drift/recovery, stability under pressure, persistence of beliefs/claims), this survey provides a **memory-operations vocabulary** that can help describe:
  - When the model is effectively *updating* vs merely *retrieving* prior context
  - Whether failures look like missing *indexing/retrieval* vs harmful *updating/consolidation*
  - How “condensation” (summaries) may create systematic drift if treated as ground truth

## How is this different from GALILEO?

- This is a **broad survey/taxonomy**, not a new protocol, benchmark, or intervention.
- Focuses on “memory as an agent capability” broadly (including parametric edits), rather than GALILEO-style behavioral evaluation of robustness dynamics under conversational pressure (if that is GALILEO’s focus).

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO has controlled protocols that separate *evidence-driven revision* from *pressure-driven drift* (and measures recovery), that would be a sharper behavioral story than survey-level categorizations.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently discusses “memory” without an explicit decomposition, reviewers may ask for clearer distinctions between:
  - retrieval vs updating
  - summarization/condensation vs consolidation
  - forgetting as deletion vs being overwritten by later context

## Action items for GALILEO (experiments / method / writing)

- [ ] In related work / background, introduce the **six operations** as a framing device, then position GALILEO as primarily targeting (which?) operations (likely updating/forgetting/retrieval under pressure).
- [ ] When presenting failure modes, label them using the ops vocabulary (e.g., “retrieval failure” vs “harmful update”).
- [ ] Consider adding at least one ablation that toggles “condensation” (summary memory) to see if it increases drift.

## Quotes / details to potentially cite

- Memory representations: **parametric** (in weights) vs **contextual** (explicit external memory).
- Six operations: **Consolidation, Updating, Indexing, Forgetting, Retrieval, Condensation**.
- Four topics mapped by the taxonomy: **long-term**, **long-context**, **parametric modification**, **multi-source memory**.
