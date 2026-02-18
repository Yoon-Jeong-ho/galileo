# LoCoMo-Plus: Beyond-Factual Cognitive Memory Evaluation Framework for LLM Agents

- Year: 2026
- Venue: arXiv
- Authors: Yifei Li; Weidong Guo; Lingling Zhang; Rongman Xu; Muye Huang; Hui Liu; Lijiao Xu; Yu Xu; Jun Liu
- URL: https://arxiv.org/abs/2602.10715
- BibTeX key (if we add it): Li2026LoCoMoPlus
- Tags: memory, long-term, conversational-agents, evaluation, benchmark, constraints, cognitive-memory

## One-sentence takeaway

LoCoMo-Plus extends long-context conversational memory evaluation beyond factual recall by testing whether an agent preserves *latent user constraints* (state/goals/values/causal constraints) across “cue–trigger semantic disconnect” and scoring *constraint consistency* rather than string overlap.

## What problem does it solve?

- Existing long-term memory benchmarks for dialogue agents largely evaluate *explicit factual recall* (earlier fact → later semantically aligned question), which misses many realistic failures where the correct response depends on *implicit constraints* inferred earlier (e.g., user goals/values) that are not explicitly re-queried later.
- Common evaluation practices (task-type disclosed prompting; BLEU/ROUGE/EM/F1-style string matching) can be misleading in open-ended response settings.

## What is the core method / protocol?

- Benchmark: **LoCoMo-Plus** (building on LoCoMo) designed to probe **Level-2 “cognitive memory”**.
- Key stressor: **cue–trigger semantic disconnect** — the earlier “cue” that implies a constraint and the later “trigger” query are not lexically/semantically aligned, so simple retrieval-by-similarity is less effective.
- Memory is framed as retaining/applying **latent constraints** (paper describes four types):
  - causal constraints
  - state constraints
  - goal constraints
  - value constraints
- Evaluation: proposes a unified framework based on **constraint consistency** (i.e., whether the response respects the latent constraint) rather than surface-form overlap.

## What are the key metrics?

- Constraint-consistency-based evaluation (primary; intended to replace/augment overlap metrics in this setting).
- Paper argues BLEU/ROUGE/EM/F1 are misaligned for cognitive-memory instances (open-ended but constraint-bound responses).

## What are the main results?

- Across multiple backbone LLMs plus retrieval-based methods / memory systems, **cognitive memory remains challenging** under cue–trigger disconnect and reveals failures not captured by factual-recall-style benchmarks.
- (High-level takeaway from the paper’s summary/abstract: improvements that look good under overlap metrics do not necessarily translate to constraint-consistent behavior.)

## How is this similar to GALILEO?

- Same overall target: **long-term agent memory** that is useful for *behavioral consistency* over extended interactions.
- Highlights that “memory” should include **persisting user constraints** (preferences, goals, values) and using them at decision time, not just answering trivia about the past.

## How is this different from GALILEO?

- Primarily an **evaluation benchmark + metric framework**, not a new memory mechanism.
- Uses a benchmark construction notion (cue–trigger semantic disconnect) that explicitly tries to defeat naive similarity retrieval; GALILEO may need to demonstrate robustness to this kind of disconnect.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO provides an end-to-end memory *system* (not just evaluation), it can position itself as a concrete method that can be stress-tested on LoCoMo-Plus.

## Where GALILEO is weaker / needs to improve

- If GALILEO evaluations rely on factual QA or surface metrics, this paper is a strong argument to add **constraint-consistency** style evaluation (and/or cognitive-memory test sets) to avoid over-claiming.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add LoCoMo-Plus as a benchmark (or at least adopt its cognitive-memory split) to the evaluation section.
- [ ] Introduce a “cue–trigger semantic disconnect” stress test: measure performance as disconnect increases (e.g., topical drift, distance, paraphrase mismatch).
- [ ] For evaluations with open-ended responses, report a constraint-consistency metric (LLM-judge or rubric-based) and explicitly discuss why overlap metrics are insufficient.
- [ ] When presenting memory retrieval components, include analyses where retrieval-by-similarity fails but constraint retention should still succeed.

## Quotes / details to potentially cite

- “Existing benchmarks … largely equate conversational memory with explicit factual recall … [but] realistic interactions … depend on implicit constraints such as user state, goals, or values …” (Intro/Abstract; paraphrase)
- “cue–trigger semantic disconnect” (their term for the key benchmark stressor)
- Cognitive memory decomposed into latent constraints: “causal, state, goal, and value.” (Problem framing)
