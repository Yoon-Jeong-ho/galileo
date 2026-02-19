# Retrieval-Augmented Generation: A Comprehensive Survey of Architectures, Enhancements, and Robustness Frontiers

- Year: 2025
- Venue: arXiv preprint (notes: “under review at ACM TOIS” per HTML)
- Authors: Chaitanya Sharma et al. (author list not captured via lightweight fetch)
- URL: https://arxiv.org/abs/2506.00054 (HTML: https://arxiv.org/html/2506.00054v1)
- BibTeX key (if we add it): sharma2025rag_survey
- Tags: rag; survey; taxonomy; robustness; evaluation; retrieval; reranking; filtering

## One-sentence takeaway

A broad 2025 survey that proposes a taxonomy of RAG architectures (retriever-/generator-/hybrid-/robustness-oriented) and synthesizes techniques for retrieval, filtering, efficiency, reranking, and robustness/security, plus common trade-offs and open problems.

## What problem does it solve?

- Consolidates a fast-moving RAG literature into a single taxonomy + narrative: what design knobs exist (retriever vs generator vs joint), what failure modes recur (noise, hallucination, latency), and how to evaluate systems/benchmarks.
- Helps position new RAG systems by mapping contributions to specific pipeline components (query rewriting, reranking, context filtering/compression, decoding control, robustness/security).

## What is the core method / protocol?

- Survey + taxonomy rather than a new algorithm.
- Taxonomy buckets:
  - Retriever-centric (query reformulation, retriever adaptation, granularity-aware retrieval)
  - Generator-centric (faithfulness-aware decoding, context compression/utility filtering, retrieval-guided generation)
  - Hybrid (multi-round retrieval, utility-driven joint optimization/RL, dynamic retrieval triggering)
  - Robustness/security-oriented (noise-adaptive training, hallucination-aware decoding constraints, adversarial/poisoning defenses)
- Synthesizes “enhancement dimensions” across the pipeline: retrieval, filtering, efficiency, robustness, reranking.

## What are the key metrics?

- As a survey, it refers to common downstream QA metrics rather than defining new ones:
  - Short-form QA: EM / F1
  - Retrieval/ranking: MRR, nDCG, Recall@k
  - Faithfulness / hallucination: task- and benchmark-specific hallucination/grounding scores
  - Efficiency: latency (e.g., TTFT), context length, compute overhead

## What are the main results?

- Provides a structured map of methods and claimed strengths/limitations (e.g., filtering reduces hallucination but can drop recall; hybrid agentic/dynamic retrieval improves multi-hop but increases latency/complexity).
- Highlights recurring trade-offs:
  - retrieval precision vs generation flexibility
  - efficiency vs faithfulness
  - modularity vs coordination
- Identifies open challenges: adaptive retrieval architectures, real-time retrieval integration, structured reasoning over multi-hop evidence, privacy-preserving retrieval.

## How is this similar to GALILEO?

- If GALILEO is a RAG-style system/paper, this is directly adjacent “related work scaffolding”:
  - same core pipeline concerns (retrieval quality, reranking/filtering, grounding/faithfulness, robustness, evaluation)
  - offers a vocabulary (taxonomy + enhancement axes) to describe GALILEO’s design choices and contribution.

## How is this different from GALILEO?

- Not a new method or benchmark; it’s a survey.
- Doesn’t provide a single, reproducible experimental setup; instead it aggregates results across many papers and tasks.

## Where GALILEO is stronger / cleaner (if true)

- GALILEO can likely be stronger on: a single crisp problem statement, a concrete method, and controlled experiments/ablation—things a survey cannot provide.

## Where GALILEO is weaker / needs to improve

- If GALILEO lacks broad positioning, this survey suggests specific axes to cover explicitly:
  - which taxonomy bucket(s) it belongs to
  - which enhancement dimensions it improves (retrieval/filtering/efficiency/robustness/reranking)
  - which trade-off it targets and why.

## Action items for GALILEO (experiments / method / writing)

- [ ] Use this taxonomy to structure the related-work section: start with the 4 buckets, then drill into the bucket most relevant to GALILEO.
- [ ] Add an “enhancement axes” paragraph/table: specify what GALILEO changes (retrieval? reranking? decoding control?) and what it keeps fixed.
- [ ] In evaluation, explicitly report at least one metric per axis relevant to your claims: retrieval quality (MRR/nDCG), end-task (EM/F1), faithfulness, and latency/context length.
- [ ] If robustness is a selling point, include at least one noise/adversarial retrieval stress test (even a simple perturbation protocol) and report degradation.

## Quotes / details to potentially cite

- Abstract-level positioning (good for an opening sentence): RAG improves grounding but introduces challenges in “retrieval quality, grounding fidelity, pipeline efficiency, and robustness against noisy or adversarial inputs.”
- Taxonomy sentence (usable as framing): architectures categorized into “retriever-centric, generator-centric, hybrid, and robustness-oriented designs.”
