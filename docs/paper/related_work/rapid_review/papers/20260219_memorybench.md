# MemoryBench: A Benchmark for Memory and Continual Learning in LLM Systems

- Year: 2025
- Venue: arXiv
- Authors: Qingyao Ai, Yichen Tang, Changyue Wang, Jianming Long, Weihang Su, Yiqun Liu
- URL: https://arxiv.org/abs/2510.17281
- BibTeX key (if we add it): a i2025memorybench
- Tags: memory, continual-learning, benchmark, llm-systems, user-feedback-simulation

## One-sentence takeaway

MemoryBench proposes a large, multi-domain/multi-task benchmark that *simulates explicit+implicit user feedback logs* to evaluate whether LLM systems can actually improve over time (procedural memory / continual learning), and finds current “memory LLM systems” are often no better than simple RAG.

## What problem does it solve?

- Existing “LLM memory” benchmarks mostly test long-context reading-comprehension style retrieval (static evaluation), not whether systems can *learn from feedback during service*.
- Lack of a comprehensive benchmark that (a) includes different memory types (declarative vs procedural), (b) includes feedback logs, and (c) evaluates continual improvement rather than one-shot recall.

## What is the core method / protocol?

- Defines a taxonomy:
  - Memory: **Declarative** (semantic + episodic) vs **Procedural** (task-execution know-how / learning from outcomes).
  - Feedback: **Explicit** (verbose critique; action feedback like like/dislike) vs **Implicit** (behavior signals like copy/exit/refined prompt).
- Builds a simulation framework with three modules:
  1) **Task Provider**: provides query q, context/corpus c (optional), and evaluation metadata v.
  2) **User Simulator**: generates feedback logs S on training cases via an **LLM-as-user** (plus programmable action simulator).
  3) **Performance Monitor**: evaluates on held-out test cases using native metrics; merges multi-metric datasets via an **LLM-as-judge** aggregation (1–10), then normalizes and averages.
- Data: 11 public datasets, spanning domains (open, legal, academic), languages (en/zh), and input/output regimes (LiSo/LiLo/SiLo/SiSo; threshold 600 tokens).
- Evaluation settings:
  - **Off-policy**: pre-generate feedback logs, then test learning from them.
  - **On-policy**: interact and update online (reported only when methods are fast enough).

## What are the key metrics?

- Uses each dataset’s official metrics (e.g., F1, accuracy, Rouge-L, METEOR/BERTScore, etc.).
- For multi-metric datasets, aggregates via **LLM-as-judge** to a single 1–10 score.
- Reports partitioned results by domain and by task format; also reports **time per case** split into “memory time” vs “predict time”.

## What are the main results?

- Simulated feedback appears *useful* in that even vanilla LLM performance can improve when feedback is provided (per their ablations).
- Across partitions, advanced memory LLM systems tested (A-Mem, Mem0, MemoryOS) are **often inconsistent** and frequently **not better than** naive RAG baselines that treat context+feedback as retrieval corpus.
- Efficiency is a major bottleneck: memory construction/inference time can be large and unstable across task formats (e.g., MemoryOS heavy memory time; Mem0 anomalous slowdowns in some settings).
- Core diagnosis: many systems effectively treat everything as declarative memory, and fail to interpret feedback logs as **procedural** knowledge.

## How is this similar to GALILEO?

- Shares the central motivation that *real deployments are interactive* and systems should be evaluated on their ability to improve based on interaction traces/feedback, not just static one-shot performance.
- Provides concrete vocabulary for separating **declarative vs procedural** information sources, which parallels GALILEO’s need to separate “context evidence” vs “pressure/interaction effects” (even if the task focus differs).

## How is this different from GALILEO?

- MemoryBench is primarily a **benchmark + simulation/eval harness** for continual learning from feedback logs across tasks, not a targeted analysis of multi-turn persuasion/pressure-driven drift.
- Focuses on system improvement over train/test splits with feedback logs, rather than fine-grained turn-level dynamics like flip timing, recovery trajectories, or pressure operators.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO’s protocol cleanly pairs “pressure-only” vs “evidence-bearing correction” conditions, it can isolate causal mechanisms of drift/revision more directly than a broad benchmark that mixes many task types.
- GALILEO can report richer *trajectory* metrics (time-to-failure, recovery/oscillation structure) that MemoryBench largely abstracts away.

## Where GALILEO is weaker / needs to improve

- If GALILEO claims generality about “learning from interaction/feedback,” MemoryBench is a reminder we should justify scope: broad task diversity, multiple languages, and multiple IO regimes matter.
- GALILEO may need a clearer mapping of what counts as “procedural feedback logs” vs “declarative context” in our own datasets.

## Action items for GALILEO (experiments / method / writing)

- [ ] In related work, cite MemoryBench as evidence that “memory LLM systems” often reduce to RAG-like retrieval, and that *procedural feedback utilization* remains an open gap.
- [ ] Consider adopting their declarative/procedural + explicit/implicit feedback taxonomy to sharpen GALILEO’s terminology (even if our feedback type is different).
- [ ] If relevant, add a short paragraph contrasting: (i) benchmarked continual learning from feedback logs (MemoryBench) vs (ii) our controlled multi-turn drift protocol.

## Quotes / details to potentially cite

- Motivation gap: existing memory benchmarks “focus on evaluating the system on homogeneous reading comprehension tasks with long-form inputs rather than testing their abilities to learn from accumulated user feedback in service time.”
- Taxonomy claim: MemoryBench provides “all types of memory and feedback data” (declarative/procedural; explicit/implicit; verbose/action).
- Key negative result: advanced memory LLM systems do not consistently outperform naive RAG baselines when feedback logs are treated as corpus, suggesting current methods poorly handle procedural feedback.
