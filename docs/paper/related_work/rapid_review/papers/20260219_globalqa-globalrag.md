# Towards Global Retrieval Augmented Generation: A Benchmark for Corpus-Level Reasoning

- Year: 2025
- Venue: arXiv (cs.CL, cs.AI)
- Authors: Qi Luo, Xiaonan Li, Tingshuo Fan, Xinchi Chen, Xipeng Qiu
- URL: https://arxiv.org/abs/2510.26205v2 (HTML: https://arxiv.org/html/2510.26205v2)
- BibTeX key (if we add it): Luo2025GlobalQA
- Tags: rag, evaluation, benchmark, global-aggregation, corpus-level-reasoning, filtering, tools

## One-sentence takeaway

Introduces **GlobalQA**, a benchmark for *corpus-level* ("global") RAG aggregation tasks, and proposes **GlobalRAG** (retrieve full docs + LLM filter + symbolic aggregation tools) which substantially improves over standard RAG baselines but overall scores remain very low.

## What problem does it solve?

- Existing RAG benchmarks mainly test **local** retrieval QA: answer is contained in a small set of chunks/docs.
- Many real applications need **global / corpus-level** reasoning (e.g., compute counts, extremes, rankings across an entire collection), where top-k chunk retrieval inherently misses dispersed evidence.
- Lack of a systematic benchmark for this “global RAG” setting makes it hard to measure progress.

## What is the core method / protocol?

- **Dataset (GlobalQA):**
  - ~13k QA pairs built on a corpus of ~2k real-world resumes across 23 domains.
  - Four task types: **counting**, **extremum** (min/max), **sorting**, **top-k extraction**.
  - Construction uses a “reverse” pipeline: generate deterministic execution trajectories (set ops + retrieval steps), execute to get ground-truth answers, then template/generate natural-language questions.
  - Document set sizes per query can be large (up to 50 docs; many need >20 docs).

- **Method (GlobalRAG):** training-free multi-stage pipeline: **retrieval → filtering → aggregation**.
  - Retrieval unit emphasizes **document integrity** (paper argues fixed-size chunking breaks metadata/value association).
  - Adds an **LLM-driven relevance filter** to remove semantically-related-but-task-irrelevant docs that would waste context.
  - Uses **task-specific aggregation tools/modules** for exact computation (count/minmax/sort/top-k), motivated by LLM numerical brittleness.

## What are the key metrics?

- **Answer F1** (token-level F1 on final answer).
- **Document F1@k (D-F1@k)**: overlap of retrieved doc set vs gold doc set (coverage vs gold).

## What are the main results?

- On GlobalQA, mainstream RAG baselines perform extremely poorly; best reported baseline is around **1.51 Answer F1**.
- Their GlobalRAG improves substantially, e.g. on **Qwen2.5-14B**: **6.63 Answer F1** vs **1.51** (strongest baseline in their setup).
- Ablations suggest:
  - Retrieval granularity / preserving doc structure matters a lot.
  - Filtering helps by removing noise at larger retrieved sets.
  - Aggregation tools are necessary for stable numerical/ranking outputs.

## How is this similar to GALILEO?

- Frames retrieval-augmented systems as pipelines where **retrieval quality and context management** strongly determine end-task performance.
- Highlights that **evaluation design** (what the benchmark actually measures) can shift what “good retrieval” means.
- Emphasizes that for harder tasks, **hybrid LLM + tools** (symbolic/algorithmic modules) can be more reliable than pure generation.

## How is this different from GALILEO?

- Targets **global/corpus-level aggregation** over many documents rather than multi-hop over a few documents.
- Introduces a benchmark grounded in a resume corpus and deterministic trajectory generation; likely different domain and supervision assumptions than GALILEO.
- Proposes explicit **aggregation modules** (count/sort/top-k) rather than relying on the LLM to compute in-context.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO focuses on generalizable reasoning and robust retrieval across heterogeneous corpora, it may be less tied to a specific synthetic trajectory schema and a single corpus type.
- If GALILEO provides clearer end-to-end guarantees or simpler components, it may avoid the multi-module complexity (retrieval + filter + multiple tools) required here.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently evaluates mostly “local” QA-style tasks, it may **under-measure** corpus-level aggregation capabilities that users care about (counts/rankings across a collection).
- If GALILEO relies on the LLM to do arithmetic/ranking from retrieved text, it may suffer similar numerical brittleness and would benefit from tool-based aggregation.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add (or cite) a **global RAG** evaluation slice: counting / min-max / sorting / top-k tasks where evidence is distributed across many docs.
- [ ] Consider an explicit **retrieve → filter → aggregate** decomposition for global queries (especially numeric/statistical ones).
- [ ] When writing related work, explicitly distinguish **local vs global RAG** and motivate why top-k retrieval is structurally mismatched for global questions.
- [ ] Compare against GlobalQA/GlobalRAG as a “stress test” baseline; even a modest gain over ~1–7 F1 can be meaningful given reported difficulty.

## Quotes / details to potentially cite

- Defines “global RAG” as aggregating/analyzing information across an entire collection to derive corpus-level insights (contrasted with local chunk retrieval).
- GlobalQA task types: counting, extremum, sorting, top-k extraction.
- Reports strongest baseline ~1.51 Answer F1; GlobalRAG reaches 6.63 Answer F1 on Qwen2.5-14B.
