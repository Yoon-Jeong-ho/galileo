# LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory

- Year: 2025
- Venue: ICLR 2025 (arXiv v2: 2025-03)
- Authors: Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang, Kai-Wei Chang, Dong Yu
- URL: https://arxiv.org/abs/2410.10813
- BibTeX key (if we add it): longmemevalWu2025
- Tags: long-term-memory, chat-assistant, multi-session, benchmark, retrieval

## One-sentence takeaway

LongMemEval is a 500-question benchmark for long-term interactive memory in chat assistants and a diagnostic framework (indexing/retrieval/reading) showing simple memory-design tweaks materially improve recall and QA over long, multi-session histories.

## What problem does it solve?

- Existing “long conversation / memory” benchmarks are either too short, not representative of task-oriented user-assistant interaction, or don’t cover key long-term memory abilities needed in sustained personalized assistants.
- Lack of a holistic eval that tests: extracting salient info, reasoning across sessions, temporal reasoning, handling updates, and abstaining when info is absent.

## What is the core method / protocol?

- Benchmark construction:
  - 500 manually created questions embedded in synthetic-but-coherent multi-session user–assistant chat histories.
  - Targets 5 abilities: information extraction, multi-session reasoning, temporal reasoning, knowledge updates, abstention.
  - “Needle-in-a-haystack”-style: relevant facts are sparsely placed across a long history.
  - Provides standard scale settings (as described by authors):
    - LongMemEvalS ~115k tokens per problem.
    - LongMemEvalM ~500 sessions (~1.5M tokens).
- Analysis framework for memory-augmented assistants decomposed into 3 stages:
  - Indexing → Retrieval → Reading.
  - Plus control points around what to store (value granularity), how to key/index (key expansion), how to form queries (time-aware expansion), and how to “read” retrieved items (structured notes / Chain-of-Note).

## What are the key metrics?

- Downstream QA accuracy on the 500 questions (overall and by ability type).
- Memory recall metrics (e.g., recall@k for retrieving the relevant memory item(s)).
- Reported deltas from design changes (absolute improvements in recall@k and QA accuracy).

## What are the main results?

- Strong baselines still degrade substantially with long histories:
  - Commercial assistants / long-context LLMs show ~30% accuracy drop on memorizing info across sustained interactions (authors’ summary).
- Effective design knobs (as summarized in the paper):
  - Fact-augmented key expansion improves retrieval recall@k by +9.4% and QA accuracy by +5.4%.
  - Time-aware query/index expansion improves temporal-reasoning retrieval recall by ~+6.8% to +11.3%.
  - Better “reading” formats (Chain-of-Note / structured data) can add up to ~+10 accuracy points.

## How is this similar to GALILEO?

- Same general setting: multi-turn interaction where earlier content must be remembered, retrieved, and correctly used later.
- Highlights that “robustness” in long-horizon dialogue often bottlenecks on memory pipelines (what’s stored, how it’s retrieved, and how it’s consumed), not only the base model.

## How is this different from GALILEO?

- Focuses on memory (long-term interactive recall + updates) rather than social pressure / sycophancy / agreement robustness per se.
- Primarily a benchmark + memory-system design study; not centered on adversarial user strategies like pressure, persuasion, or consistency attacks.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO targets adversarial dynamics (e.g., pressure/stance shifts), it covers failure modes LongMemEval does not explicitly stress.

## Where GALILEO is weaker / needs to improve

- If GALILEO includes long-horizon settings, it likely needs a clearer decomposition of memory stages and explicit retrieval diagnostics (recall@k, ablations on indexing granularity, time-aware constraints).

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a memory-stage decomposition in the related-work / method framing: indexing vs retrieval vs reading, to clarify where failures occur.
- [ ] If applicable, add retrieval diagnostics alongside task success: recall@k of “needed past turn(s)” + usage correctness.
- [ ] Consider an evaluation slice inspired by LongMemEval’s 5 abilities (especially knowledge updates + abstention), since these interact with multi-turn robustness.

## Quotes / details to potentially cite

- Benchmark scope: “evaluate five core long-term memory abilities of chat assistants: information extraction, multi-session reasoning, temporal reasoning, knowledge updates, and abstention.”
- Observed degradation: “commercial chat assistants and long-context LLMs showing a 30% accuracy drop on memorizing information across sustained interactions.”
- Framework: “breaks down the long-term memory design into three stages: indexing, retrieval, and reading.”
- Optimization examples: session/round decomposition, fact-augmented key expansion, and time-aware query expansion; plus improved reading via Chain-of-Note / structured formats.
