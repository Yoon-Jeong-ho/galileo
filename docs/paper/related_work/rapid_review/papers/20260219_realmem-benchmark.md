# RealMem: Benchmarking LLMs in Real-World Memory-Driven Interaction

- Year: 2026
- Venue: arXiv
- Authors: Haonan Bian, Zhiyuan Yao, Sen Hu, Zishan Xu, Shaolei Zhang, Yifu Guo, Ziliang Yang, Xueran Han, Huacan Wang, Ronghao Chen
- URL: https://arxiv.org/html/2601.06966
- BibTeX key (if we add it): realmem2026
- Tags: memory, long-horizon, multi-session, benchmark, project-oriented

## One-sentence takeaway

RealMem is a synthetic-but-“realistic project” benchmark (2k+ cross-session dialogues, 11 scenarios) designed to test whether agent memory systems can track evolving project state and handle interleaved, natural queries—not just post-hoc factual QA.

## What problem does it solve?

- Existing long-context / memory benchmarks (e.g., LoCoMo, LongMemEval, HaluMem per the paper) overemphasize static recall or externally-posed QA after the conversation, and under-test:
  - evolving project state (plans that change over time),
  - interleaved multi-session workflows,
  - proactive alignment behaviors (inferring what to do next from remembered preferences / schedules).
- RealMem aims to provide an evaluation setting closer to “long-term project companion” assistants.

## What is the core method / protocol?

- Benchmark construction pipeline (3 stages):
  1) Project Foundation Construction: build persona + project goals + hierarchical blueprint → events → session summaries; define “project attributes” as dynamic state variables.
  2) Multi-Agent Dialogue Generation: simulate sessions with a User Agent + Assistant Agent; sessions are interleaved across multiple projects (a “SuperApp” framing). Assistant sees relevant memory + a global schedule to reduce temporal inconsistency.
  3) Memory and Schedule Management: extract memory points + schedule updates; semantic deduplication; iterate so memory evolves alongside dialogues (closed loop).
- Query types (used to probe memory use):
  - Temporal reasoning (schedule conflict / sequencing)
  - Static retrieval (continue from stable state)
  - Dynamic updating (revise existing plan under new constraints)
  - Proactive alignment (user is vague/affective; agent should leverage memory to propose next steps)
- Evaluation:
  - Retrieval metrics: Recall@k, NDCG@k.
  - LLM-based semantic metrics: “Mem Recall”, “Mem Helpful”.
  - Response quality: QA Score via a rubric focused on state-alignment (not just fluency).
  - Two context settings: memory-only (top retrieved memory items) vs session-based (top retrieved sessions).

## What are the key metrics?

- Retrieval: Recall@10/20, NDCG@10/20.
- Memory semantic eval: Mem Recall, Mem Helpful.
- Generation: QA Score (judged by GPT-4o / GPT-4o-mini in their experiments).
- Efficiency: average add-memory time, retrieve time, and token cost.

## What are the main results?

- On their benchmark, existing memory systems still struggle with long-term project state consistency; there remains a large gap to an Oracle memory upper bound.
- Reported comparisons include Mem0, A-mem, MemoryOS, Graph Memory:
  - Memory-only setting: MemoryOS strongest QA among non-oracle methods (paper’s claim).
  - With session context: Graph Memory strongest (suggesting entity-relational structure matters).
  - Retrieval vs generation: higher NDCG (ranking quality/precision) correlates more with QA than raw recall—noise hurts.
- Cost/latency tradeoffs: memory ingestion is substantially slower than retrieval; some systems incur high token overhead.

## How is this similar to GALILEO?

- Same high-level target: long-horizon assistants/agents that must maintain coherent state across sessions.
- Emphasizes *project state* (plans, schedules, progress) rather than isolated facts—aligned with “agent as ongoing collaborator” framing.
- Highlights proactive behaviors (alignment/clarification) and temporal constraints (scheduling), which are common requirements for real deployments.

## How is this different from GALILEO?

- Primarily a benchmark + synthesis pipeline paper, not a new memory mechanism.
- Uses a simulated multi-agent generation setup (User Agent / Assistant Agent + manager agents) to create data; may diverge from real human behavior.
- Evaluation is centered on memory retrieval and state-aligned response scoring, rather than (unclear here) tool use / environment interaction; paper notes tool-use evaluation is not yet included.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO includes real task execution/tool use, RealMem’s current scope (memory-centric; no tool-use evaluation yet) is narrower.
- If GALILEO uses real user logs or stronger grounding, it may avoid simulation artifacts inherent in multi-agent data generation.

## Where GALILEO is weaker / needs to improve

- If GALILEO lacks an evaluation suite covering (a) interleaved projects, (b) dynamic state updates, and (c) proactive alignment, RealMem’s taxonomy is a good checklist for coverage.
- If GALILEO reports only recall-like metrics, RealMem argues precision/ranking (NDCG) and usefulness metrics better predict downstream answer quality.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add/align evaluation slices by query type: static retrieval vs dynamic updating vs proactive alignment vs temporal reasoning.
- [ ] When reporting retrieval, include NDCG@k alongside Recall@k; analyze noise sensitivity.
- [ ] Consider an “Oracle memory” ablation to quantify the headroom and separate retrieval errors from generation errors.
- [ ] If applicable, add an efficiency section: memory write latency vs retrieval latency vs token cost.

## Quotes / details to potentially cite

- Abstract-level framing: “existing benchmarks primarily focus on casual conversation or task-oriented dialogue, failing to capture long-term project-oriented interactions where agents must track evolving goals.”
- Benchmark scale claim: “over 2,000 cross-session dialogues across eleven scenarios.”
- Distinguishing features (Table 1 in the paper): natural user queries, interleaved QA timing, proactive alignment, project state memory.
