# Memory in the Age of AI Agents

- Year: 2026
- Venue: arXiv (survey)
- Authors: Yuyang Hu, Shichun Liu, Yanwei Yue, Guibin Zhang, …, Shuicheng Yan (many authors)
- URL: https://arxiv.org/abs/2512.13564
- BibTeX key (if we add it): hu2026memory-age-ai-agents
- Tags: agents, memory, survey

## One-sentence takeaway

A broad survey that tries to un-fragment “agent memory” by proposing a 3-axis conceptual framework (forms / functions / dynamics) plus a compilation of benchmarks and open-source memory frameworks.

## What problem does it solve?

- Agent-memory research is growing quickly but is conceptually messy: inconsistent terminology, heterogeneous implementations, and incomparable evaluations.
- Standard “short-term vs long-term memory” taxonomies don’t adequately describe current agent memory systems.

## What is the core method / protocol?

- Survey + taxonomy construction.
- Scope clarification: distinguishes “agent memory” from adjacent concepts (LLM memory, RAG, context engineering).
- Proposes three complementary lenses:
  - **Forms**: token-level memory, parametric memory, latent memory.
  - **Functions**: factual, experiential, working memory.
  - **Dynamics**: how memory is formed, evolves, and is retrieved over time.
- Summarizes memory benchmarks and open-source frameworks; discusses forward-looking directions (automation, RL integration, multimodal, multi-agent, trustworthiness).

## What are the key metrics?

- Not a single metric paper; it surveys benchmark families and evaluation practices.
- Practical “metric” contribution is the **categorization** of what should be measured across different memory functions/dynamics (e.g., formation/update/retrieval, forgetting, trustworthiness).

## What are the main results?

- A “unified” conceptual map of agent memory that aims to help compare systems across:
  - where memory lives (tokens/weights/latents),
  - what it is for (facts vs experience vs working state),
  - how it changes over time (update/decay/retrieval policies).
- A consolidated overview of benchmarks + tooling/framework ecosystem (useful as a reference section anchor in related work).

## How is this similar to GALILEO?

- If GALILEO is positioned as an **agentic system** (or evaluates agent behavior), this survey is directly relevant as:
  - shared vocabulary for describing memory components,
  - a map of benchmark space for “memory over time”, and
  - a checklist of trustworthiness issues (memory poisoning, erroneous persistence, privacy).

## How is this different from GALILEO?

- This is a **survey / taxonomy** paper, not a new algorithm or evaluation protocol.
- It likely won’t provide the kind of tight, controlled experimental methodology that GALILEO aims for (depending on GALILEO’s contribution).

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO contributes a concrete evaluation protocol (e.g., controlled perturbations, time-to-failure, recovery metrics), it can be positioned as a more *operationalized* counterpart to this survey’s conceptual framing.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently lacks a clear “memory taxonomy” section, this paper is a strong anchor to justify terminology and to separate memory from RAG/context engineering.

## Action items for GALILEO (experiments / method / writing)

- [ ] In related work, cite this survey for: (i) why “short vs long-term” is insufficient, and (ii) the forms/functions/dynamics lenses.
- [ ] Use the survey’s terminology to label GALILEO’s target phenomena (e.g., factual vs experiential vs working memory) and to motivate which benchmark gaps GALILEO addresses.
- [ ] Cross-check the survey’s benchmark table(s) to ensure GALILEO cites the closest memory-evaluation neighbors (especially long-horizon + update/forgetting + trustworthiness).

## Quotes / details to potentially cite

- “Traditional taxonomies such as long/short-term memory have proven insufficient to capture the diversity of contemporary agent memory systems.”
- The proposed three axes:
  - forms: “token-level, parametric, and latent memory”
  - functions: “factual, experiential, and working memory”
  - dynamics: memory is “formed, evolved, and retrieved over time”
