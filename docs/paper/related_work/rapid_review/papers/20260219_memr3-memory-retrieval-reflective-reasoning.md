# MemR3: Memory Retrieval via Reflective Reasoning for LLM Agents

- Year: 2025
- Venue: arXiv (submitted to ICML per paper)
- Authors: Xingbo Du et al.
- URL: https://arxiv.org/abs/2512.20237
- BibTeX key (if we add it): memr3_du_2025
- Tags: agent-memory, conversational-memory, closed-loop-retrieval, reflection, controller, langgraph, evidence-gap

## One-sentence takeaway

MemR3 wraps an existing long-term memory store with a simple closed-loop controller (retrieve/reflect/answer) driven by an explicit evidence–gap state, improving LoCoMo QA over standard retrieve-then-answer.

## What problem does it solve?

- Long-term conversational memory systems often focus on storing/compressing memories, but retrieval is typically an open-loop heuristic (retrieve once, then answer), leading to under-retrieval (missing key facts) or over-retrieval (noise/latency).
- The paper targets *control* of retrieval: deciding when to retrieve again, when to reflect, and when to stop and answer.

## What is the core method / protocol?

- Implement MemR3 as a controller program (LangGraph) that routes among three actions:
  - **Retrieve**: call an existing backend retriever (e.g., chunk-based RAG; graph-based Zep), potentially multiple rounds with refined queries.
  - **Reflect**: reason over currently collected evidence, identify what is still missing, and rewrite/target the next retrieval.
  - **Answer**: produce final answer from accumulated evidence.
- Maintain a global **evidence–gap tracker** state (E,G):
  - E = what has been established/collected from memory
  - G = what requirements/information are still missing to answer the question
- The evidence–gap state is used to (i) refine retrieval queries, (ii) provide early stopping, and (iii) expose an explainable trace.

## What are the key metrics?

- LoCoMo benchmark performance.
- LLM-as-a-Judge score is highlighted.
- Reported relative improvements when MemR3 is used as a controller on top of existing backends.

## What are the main results?

- On LoCoMo, MemR3 surpasses strong baselines on LLM-as-a-Judge.
- Reported improvements over underlying retrievers (paper highlights):
  - RAG backend: +7.29% (overall)
  - Zep backend: +1.94% (overall)
- Qualitatively: avoids “retrieve once then hallucinate/miscalculate” by iterating retrieve↔reflect until a specific missing fact is found.

## How is this similar to GALILEO?

- Treats memory use as *agentic control* rather than a fixed pipeline.
- Emphasizes transparency/traceability of “what evidence supports the answer” via an explicit intermediate state.
- Backend-agnostic wrapper around an existing store/retriever (plug-and-play controller framing).

## How is this different from GALILEO?

- Focus is specifically on long-horizon conversational memory QA (LoCoMo) and retrieval control loops, not broader planning/execution.
- Uses a relatively simple action space (retrieve/reflect/answer) and an explicit evidence–gap bookkeeping abstraction.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO already has stronger task decomposition/planning beyond retrieval (tools, environment actions), it may generalize beyond conversational memory QA.

## Where GALILEO is weaker / needs to improve

- If GALILEO retrieval is still largely open-loop (single retrieval then generate), MemR3-style evidence–gap tracking + controller routing is a clear pattern to adopt.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding an explicit evidence–gap state to retrieval episodes (what we know vs what we still need), and use it to decide: retrieve again vs stop.
- [ ] Add “closed-loop retrieval controller” positioning in related work: distinguish storage/compression vs retrieval control.
- [ ] If we have a long-horizon memory benchmark, compare open-loop retrieve-then-answer vs closed-loop retrieve↔reflect with early stopping.

## Quotes / details to potentially cite

- “...build memory retrieval as an autonomous, accurate, and compatible agent system... MemR3 ... router that selects among retrieve, reflect, and answer actions ... global evidence-gap tracker ...” (Abstract)
- “MemR3 maintains a global evidence–gap state (E,G) that summarizes what has been reliably established ... and what information remains missing.” (Intro)
- “...improves existing retrievers ... overall improvement on RAG (+7.29%) and Zep (+1.94%) ...” (Abstract)
