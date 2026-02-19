# Frustratingly Simple Retrieval Improves Challenging, Reasoning-Intensive Benchmarks

- Year: 2025
- Venue: arXiv
- Authors: Xinxi Lyu; Michael Duan; Rulin Shao; Pang Wei Koh; Sewon Min
- URL: https://arxiv.org/abs/2507.01297
- BibTeX key (if we add it): lyu2025frustratingly
- Tags: rag, retrieval, datastore, dense-retrieval, ann, diskann, reasoning-benchmarks, mmlu, gpqa, math

## One-sentence takeaway

A minimal dense-retrieval RAG setup can meaningfully improve reasoning benchmarks if (and mostly only if) the datastore is web-scale but aggressively quality-filtered and served with a hybrid ANN-then-exact search stack.

## What problem does it solve?

- Prior RAG results on reasoning-intensive benchmarks (MMLU/MMLU-Pro/GPQA/MATH) often show little or negative benefit, leading to a belief that retrieval is mainly for factoid QA.
- The paper argues the real blocker is not the RAG recipe, but the lack of an accessible, high-coverage datastore aligned with pretraining breadth (Wikipedia-only is too narrow; naive web-scale is too noisy and too expensive to serve).

## What is the core method / protocol?

- Build an in-house datastore (CompactDS) intended to match pretraining breadth but remain practically servable on a single node.
- Key design choices:
  - Data curation: start from web crawl (Common Crawl), then aggressively filter to a high-quality subset (e.g., union of C4 + DCLM-Baseline; additional filtering with FineWeb-Edu classifier thresholding) and add diverse high-quality sources (Wikipedia variants, books, educational text, curated math, academic papers).
  - Retrieval serving: combine in-memory approximate nearest neighbor (ANN) retrieval with an on-disk exact inner-product search stage (cited as enabling good speed/recall tradeoff).
- Evaluate a minimal pipeline: dense retrieval -> stuff retrieved chunks into context -> standard generation, across multiple model sizes (8B to 70B).

## What are the key metrics?

- Accuracy on reasoning-oriented benchmarks: MMLU, MMLU Pro, AGI Eval, GPQA, MATH (also mentions subsets like GPQA Diamond, MATH-500).
- Retrieval/serving practicality: retrieval accuracy and latency (claims subsecond latency on a single node).
- Ablations over datastore composition and retrieval stack variants (ANN-only vs ANN + exact search).

## What are the main results?

- With CompactDS, minimal RAG yields consistent gains across benchmarks/model sizes.
- Reported relative gains (example with LLaMa 3.1 8B Instruct):
  - +10% on MMLU
  - +33% on MMLU Pro
  - +14% on GPQA
  - +19% on MATH
- Hybrid retrieval helps: exact inner-product search after ANN with a more expressive encoder improves over ANN alone (paper highlights widening gains on MMLU Pro and MATH).
- Datastore diversity matters: no single source is sufficient; removing even weaker sources can degrade performance.
- Claims their in-house datastore matches or outperforms Google Search (and some agentic/web-search-based systems) while staying simple and reproducible.

## How is this similar to GALILEO?

- Treats retrieval quality/coverage as a first-class lever for downstream reasoning performance (not just prompt engineering).
- Emphasizes reproducibility and self-contained retrieval (in-house corpus + deterministic serving), aligning with systems-oriented work.
- Uses strong ablation framing: performance depends on datastore construction choices and retrieval stack.

## How is this different from GALILEO?

- Focus is primarily on datastore construction + retrieval serving efficiency (ANN + exact), rather than novel reasoning-time control, planning, or interaction protocols (if GALILEO has those).
- Evaluates a “minimal RAG” baseline rather than multi-step/agentic reasoning loops.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO targets task-specific structure (e.g., structured retrieval, iterative evidence integration, or calibrated citation), it may deliver clearer interpretability than single-shot “stuffing” RAG.

## Where GALILEO is weaker / needs to improve

- If GALILEO relies on smaller/narrower corpora (e.g., Wikipedia-only), this paper suggests it will underperform on broad reasoning benchmarks due to coverage mismatch.
- If GALILEO does not use a hybrid ANN+exact second stage, it may leave accuracy on the table at the same latency budget.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a related-work paragraph: “retrieval can help reasoning benchmarks when the datastore is broad and high-quality; failures are often datastore-limited.”
- [ ] If we are building/using an in-house datastore, report its composition and filtering explicitly; run ablations on (i) quality filtering and (ii) diversity of sources.
- [ ] Try a two-stage retrieval stack (ANN shortlist + exact rerank/search) and measure accuracy vs latency tradeoffs.
- [ ] Add a baseline comparison against web-search-powered RAG (if feasible) but emphasize reproducibility constraints.

## Quotes / details to potentially cite

- “We introduce CompactDS: a diverse, high-quality, web-scale datastore that achieves high retrieval accuracy and subsecond latency on a single-node.”
- “CompactDS, a 380-billion-word datastore … combining in-memory approximate nearest neighbor (ANN) search with on-disk exact inner product search enables subsecond retrieval on a single 456GB RAM node.”
- Reported relative gains with minimal RAG: “+10% on MMLU, +33% on MMLU Pro, +14% on GPQA, and +19% on MATH.”
