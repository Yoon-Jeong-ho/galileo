# Comprehensive Comparison of RAG Methods Across Multi-Domain Conversational QA

- Year: 2026
- Venue: EACL SRW 2026 (arXiv)
- Authors: Klejda Alushi, Jan Strich, Chris Biemann, Martin Semmann
- URL: https://arxiv.org/abs/2602.09552
- BibTeX key (if we add it): Alushi2026ComprehensiveRAGConvQA
- Tags: conversational-qa, rag, retrieval, reranking, hyde, hybrid-bm25, benchmarking

## One-sentence takeaway

Across 8 conversational QA datasets, simple retrieval upgrades (reranking, hybrid BM25, HyDE) reliably beat vanilla RAG, while several “advanced” RAG variants fail to help and can underperform even No-RAG.

## What problem does it solve?

- Lack of systematic, apples-to-apples comparison of common RAG variants in *multi-turn* conversational QA.
- Prior work often evaluates methods in isolation and/or focuses on single-turn QA, missing conversational complications (history dependence, coreference, topic/intent drift across turns).

## What is the core method / protocol?

- Unified benchmark study over 8 subsets of ChatRAG-Bench (derived from ChatQA): QuAC, SQA, QReCC, TopiOCQA, Doc2Dial, DoQA, CoQA, INSCIT.
- Two reference points:
  - **No RAG**: generator uses only dialogue history.
  - **Oracle Context**: generator gets gold context (ceiling).
- Compared retrieval/generation pipelines:
  - **Basic**: Vanilla dense RAG; BM25; Hybrid BM25 (sparse+dense); Cross-encoder reranker on initial retrieval.
  - **Advanced** (as implemented in their framework): HyDE; query rewriting; summarization; SumContext (summary + original docs); HyDE-reranker.
- Model/setup (per paper): Llama 3 8B Instruct; Chroma DB with all-MiniLM-L6-v2 embeddings; cosine similarity; temperature 0; max output 1000 tokens; long context window.
- Analysis includes turn-position effects (how retrieval metrics change across conversation turns).

## What are the key metrics?

- **Retriever**: MRR@5 (main), Recall@k (reported in appendix).
- **Generator**: token-level **F1** (SQuAD-style; max over references when multiple answers).
- Also discusses correlation between retriever quality (MRR) and generation quality (F1) across datasets/methods.

## What are the main results?

- **HyDE** is often best on retrieval (best on 5/8 datasets by MRR@5) and strong on generation; for INSCIT, HyDE reportedly *triples* MRR vs vanilla RAG.
- **Hybrid BM25** yields consistently higher F1 than vanilla RAG across datasets.
- **Reranker** is among the top methods for final answer quality (F1), with HyDE/Hybrid BM25 also strong.
- Several “advanced” techniques (notably some rewriting/summarization variants) are **highly dataset-dependent** and can **degrade** performance—sometimes below the No-RAG baseline.
- Turn-depth effects are mixed:
  - Datasets with high context-to-question ratios and topic switching (e.g., TopiOCQA/INSCIT) tend to degrade with more turns.
  - More stable-context datasets (e.g., CoQA/SQA) can improve across turns.
- High retrieval quality does **not** guarantee high F1 (e.g., DoQA shows very high MRR but comparatively low F1), suggesting generator/answer-format bottlenecks.

## How is this similar to GALILEO?

- Emphasizes **robust evaluation** and **turn-aware analysis** rather than showcasing a single method.
- Highlights that conversational settings require careful handling of **history**, **coreference**, and **topic drift**—likely overlapping with GALILEO’s multi-turn / interaction focus.
- Provides evidence that relatively “simple” retrieval interventions (hybrid sparse+dense, reranking) can be strong baselines that GALILEO should compare against.

## How is this different from GALILEO?

- Primarily a **benchmarking/comparison** paper; it does not propose a new conversational-RAG algorithm.
- Uses specific infra choices (EncouRAGe, Chroma, MiniLM embeddings, Llama 3 8B) and focuses on QA datasets; GALILEO may target different tasks, components, or stronger agentic/iterative protocols.
- Evaluates mostly *single-shot* retrieval+generation variants; less focus on multi-step/agentic retrieval planning (if GALILEO includes that).

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO includes explicit mechanisms for turn-level uncertainty, history selection, or adaptive retrieval policies, it can go beyond the “static method menu” compared here.
- If GALILEO uses more faithful citations/attribution or stronger human-eval, it could address limitations of F1-only evaluation.

## Where GALILEO is weaker / needs to improve

- Must ensure baselines include **hybrid sparse+dense** and **reranking**, since these are repeatedly strong and not “exotic.”
- Needs to report **retriever metrics** alongside generator metrics (MRR/Recall) and examine turn-depth effects; otherwise the story can be incomplete.

## Action items for GALILEO (experiments / method / writing)

- [ ] Include Hybrid BM25 + dense retrieval baseline, and a cross-encoder reranker baseline, in the main table.
- [ ] Add turn-position breakdown plots/tables (MRR@k and task metric by turn) to show robustness under longer dialogues / topic switching.
- [ ] Add analysis section discussing when “advanced” methods (rewriting/summarization) hurt, and relate to dataset structure (Ctx/Q ratio, topic switching).
- [ ] If GALILEO uses any rewriting/summarization, add an ablation showing when it drops below No-RAG and why.

## Quotes / details to potentially cite

- Abstract-level claim: robust methods like **reranking**, **hybrid BM25**, and **HyDE** “consistently outperform vanilla RAG,” while some advanced methods can “degrade performance below the No-RAG baseline.”
- Key framing: conversational QA complicates retrieval due to “dialogue history, coreference, and shifting user intent.”
- Practical conclusion: effectiveness depends less on “method complexity” and more on alignment between retrieval strategy and dataset structure.
