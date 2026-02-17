# Retrieval-Augmented Generation: A Comprehensive Survey of Architectures, Enhancements, and Robustness Frontiers

- Year: 2025
- Venue: arXiv (preprint; under review at ACM TOIS per arXiv HTML)
- Authors: Chaitanya Sharma
- URL: https://arxiv.org/abs/2506.00054
- BibTeX key (if we add it): sharma2025rag_survey
- Tags: rag, survey, robustness, evaluation

## One-sentence takeaway

A broad, taxonomy-driven survey of RAG architectures and “enhancement knobs” (retrieval/filtering/reranking/decoding/efficiency), with a dedicated robustness/security lens (noise, hallucination control, poisoning/backdoors) and pointers to evaluation trends.

## What problem does it solve?

- Consolidates a fast-moving RAG literature into a single taxonomy + structured map of design choices.
- Frames RAG not only as “factuality improvement” but as a system with new failure modes: noisy retrieval, grounding drift, latency/efficiency, and adversarial retrieval/corpus attacks.

## What is the core method / protocol?

- Survey + taxonomy:
  - Retriever-centric vs generator-centric vs hybrid vs robustness/security-oriented architectures.
- Organizes “enhancements” across the pipeline:
  - retrieval strategies (query rewriting/decomposition, multi-source, structured/graph retrieval)
  - context filtering / compression
  - decoding control / faithfulness-aware generation
  - efficiency (caching, pipelining, sparse selection)
  - reranking methods (adaptive, unified, fusion)
- Discusses evaluation directions (retrieval-aware eval, robustness tests, federated retrieval) and recurring trade-offs.

## What are the key metrics?

(As a survey, it reports common metrics rather than introducing a new one.)

- QA-style answer quality: EM / F1 (esp. short-form + multi-hop QA)
- Retrieval quality: Recall@K / MRR / nDCG
- Faithfulness / hallucination: task-specific hallucination or grounding metrics/benchmarks (referenced as a trend)
- Efficiency: latency / time-to-first-token (TTFT), context length, compute overhead
- Robustness: performance under noisy/irrelevant/adversarially-poisoned retrieval contexts

## What are the main results?

- Not a new algorithm; main “result” is the structured synthesis:
  - A 4-way taxonomy that’s useful for positioning new RAG work.
  - A robustness/security section that explicitly treats corpus poisoning/backdoors as first-class concerns for RAG deployments.
  - Identifies recurring trade-offs:
    - retrieval precision vs generation flexibility
    - efficiency vs faithfulness
    - modularity vs tight retriever–generator coordination

## How is this similar to GALILEO?

- Useful as a *related-work scaffolding* paper: it’s a taxonomy+evaluation lens for a system that combines multiple modules and exposes new failure modes.
- The robustness framing (noise/adversarial inputs, evaluation under perturbations) is conceptually aligned with “stress testing” / failure-mode analysis.

## How is this different from GALILEO?

- Domain focus is RAG architectures and retrieval–generation pipelines, not multi-turn social pressure / belief drift / sycophancy-style robustness.
- Emphasizes retrieval and grounding failures (including poisoning/backdoors) rather than interaction-driven instability.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO targets interaction-driven robustness (multi-turn drift/pressure), it can provide cleaner causal attribution than generic RAG robustness notions (which confound retriever quality, context selection, and generator behavior).

## Where GALILEO is weaker / needs to improve

- If GALILEO includes retrieval/grounding components (or claims “evidence-based” behavior), it should explicitly address RAG-specific threats surfaced here:
  - robustness to irrelevant/misleading retrieval
  - corpus poisoning / backdoor risk
  - evaluation that separates retrieval errors from generation errors

## Action items for GALILEO (experiments / method / writing)

- [ ] (Writing) Use this survey’s taxonomy to place any retrieval-augmented or tool-augmented baselines we mention; cite it as a “one-stop” overview.
- [ ] (Method) If GALILEO uses external evidence, add a small “retrieval noise” stress test: inject irrelevant or adversarial passages and measure degradation.
- [ ] (Method) Add a brief security paragraph: corpus poisoning / TrojanRAG/BadRAG-type threats as an adjacent risk model.

## Quotes / details to potentially cite

- Abstract-level positioning: RAG improves grounding but introduces challenges in “retrieval quality, grounding fidelity, pipeline efficiency, and robustness against noisy or adversarial inputs.”
- Taxonomy framing: architectures grouped into “retriever-centric, generator-centric, hybrid, and robustness-oriented designs.”
