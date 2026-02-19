# Cognitive Memory in Large Language Models

- Year: 2025
- Venue: arXiv
- Authors: Lianlei Shan (per arXiv submission page)
- URL: https://arxiv.org/abs/2504.02441
- BibTeX key (if we add it): shan2025cognitiveMemoryLLMs
- Tags: memory, survey, long-context, kv-cache, compression, rag, parameter-efficient-tuning

## One-sentence takeaway

A broad survey that frames “LLM memory” as (i) text/external-store memory, (ii) KV-cache memory selection/compression/management, (iii) parameter-based memory (e.g., LoRA/TTT/MoE), and (iv) hidden-state/recurrent long-context models.

## What problem does it solve?

- Provides a taxonomy + terminology for “memory in LLMs,” motivated by (claimed) benefits like better context-rich responses, reduced hallucination, and improved efficiency.
- Helps organize the design space across retrieval/external memory, in-context KV caching, and architecture/parameter updates.

## What is the core method / protocol?

- Not a single method; it is a structured review.
- Categorization (per abstract):
  - **Sensory memory** ≈ the raw input prompt.
  - **Short-term memory** ≈ immediate-context processing.
  - **Long-term memory** ≈ external DB / structures.
- For **text-based memory**, it summarizes pipelines:
  - acquisition: selection + summarization
  - management: update/access/store/conflict resolution
  - utilization: full-text search, SQL, semantic search
- For **KV-cache memory**, it covers:
  - selection methods: regularity-based summarization, score-based, special-token embeddings
  - compression: low-rank compression, KV merging, multimodal compression
  - management: offloading, shared-attention mechanisms
- For **parameter-based memory**, it highlights turning “memories” into parameters (LoRA, TTT, MoE).
- For **hidden-state memory**, it highlights chunking/recurrent transformers/Mamba-style state.

## What are the key metrics?

- Not specified in the abstract (survey); likely reports task-level metrics from covered works (not extracted in this rapid pass).

## What are the main results?

- Main “result” is the organization of approaches + a set of future directions (per abstract).
- No single experimental claim to cite directly (survey synthesis).

## How is this similar to GALILEO?

- GALILEO cares about **multi-turn stability/robustness across rounds**; memory mechanisms are a core ingredient in many multi-turn systems (agents) and can be a confounder or lever for drift control.
- The survey’s taxonomy (text/RAG vs KV-cache vs parameter/hidden-state) can help position GALILEO’s design choices (what we assume about persistence across turns, and what is “state”).

## How is this different from GALILEO?

- This paper is **not** focused on:
  - pressure testing / adversarial multi-turn dynamics,
  - sycophancy/persuasion,
  - belief revision vs “drift” controls,
  - explicit behavioral stability criteria.
- It is broad and mechanism-focused (systems/architecture) rather than evaluation of conversational robustness.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO provides a clear definition/benchmark for multi-turn robustness (drift, persuasion, sycophancy), that is more directly actionable than a general memory taxonomy.
- If GALILEO separates **statefulness** (memory) from **alignment/robustness** effects in evaluation, that would be a clearer contribution.

## Where GALILEO is weaker / needs to improve

- If GALILEO discusses “memory” only as RAG / external notes, this survey suggests we may need to acknowledge/contrast against KV-cache management/compression and parameter/hidden-state approaches as alternate “memory” mechanisms.
- GALILEO writing may need a short “memory mechanism landscape” paragraph to avoid reviewer pushback (“you ignore long-context / cache compression / recurrent state”).

## Action items for GALILEO (experiments / method / writing)

- [ ] Writing: add a short related-work subsection that distinguishes: external-text memory (RAG/DB), KV-cache memory, parameter-based memory, hidden-state/recurrent memory.
- [ ] Experiment design: explicitly state what “memory” the evaluated system has (persistent store? only conversation transcript? cache reuse?), to avoid ambiguity in multi-turn robustness claims.
- [ ] Threat model: note that some “stability across rounds” changes may come from memory selection/compression (KV policies), not only from the agent/policy itself.

## Quotes / details to potentially cite

- Abstract-level taxonomy bullets (paraphrased): text-based memory (acquisition/management/utilization); KV-cache memory (selection/compression/management); parameter-based memory (LoRA/TTT/MoE); hidden-state-based memory (chunking/recurrent transformers/Mamba).
