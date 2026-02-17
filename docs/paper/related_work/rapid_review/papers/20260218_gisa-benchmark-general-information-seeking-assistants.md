# GISA: A Benchmark for General Information-Seeking Assistant

- Year: 2026
- Venue: arXiv
- Authors: Yutao Zhu; Xingshuo Zhang; Maosen Zhang; Jiajie Jin; Liancheng Zhang; Xiaoshuai Song; Kangzhi Zhao; Wencong Zeng; Ruiming Tang; Han Li; Ji-Rong Wen; Zhicheng Dou
- URL: https://arxiv.org/abs/2602.08543
- BibTeX key (if we add it): zhu2026gisa
- Tags: benchmark, information-seeking, search agents, web, trajectories, evaluation

## One-sentence takeaway

GISA is a 373-query human-written benchmark for web information-seeking assistants with structured answer formats and gold human search trajectories, aiming to reduce “backward constructed” query artifacts and contamination.

## What problem does it solve?

- Existing search-agent benchmarks often:
  - build questions backward from known answers (unnatural / not representative),
  - separate “single fact lookup” vs “multi-source aggregation” rather than unifying them,
  - rely on static answer sets that are prone to memorization / data contamination.
- Need a benchmark that reflects real user information-seeking, supports deterministic scoring, and provides process supervision.

## What is the core method / protocol?

- Dataset/benchmark construction (not a model):
  - 373 human-crafted queries intended to reflect authentic information needs.
  - Four structured answer formats: **item**, **set**, **list**, **table** (designed to make evaluation more deterministic).
  - Includes a **live subset** whose answers are periodically updated to resist memorization.
  - Provides **complete human search trajectories for every query** (useful for imitation learning / process-level supervision).
- Evaluation:
  - Report exact match (EM) over structured outputs; analyze degradation on tasks needing complex planning + comprehensive gathering.

## What are the key metrics?

- Exact match (EM) on structured outputs (overall and by task type/format).
- (Implicit/likely) subset analysis on “live” vs static portions; planning/aggregation difficulty slices.

## What are the main results?

- On tested mainstream LLMs and commercial search products, best model reaches **19.30% EM**.
- Performance drops notably on tasks that require **complex planning** and **broad information gathering**.

## How is this similar to GALILEO?

- Same problem space: evaluating/understanding **multi-turn, web-facing information-seeking assistants/agents**.
- Emphasizes tasks that require both **reasoning** and **multi-source aggregation**, which matches typical agentic retrieval workloads.

## How is this different from GALILEO?

- GISA is primarily a **benchmark + trajectories** contribution (data/eval), not a new agent architecture.
- Strong emphasis on **structured answer formats** (item/set/list/table) and **deterministic EM** scoring.
- Provides **gold human trajectories for every query**, which may exceed what GALILEO currently assumes/uses for supervision.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO targets a specific, well-scoped task distribution and/or provides clearer modeling ablations, it may offer more direct methodological insight than a benchmark paper.
- If GALILEO explicitly addresses robustness/safety/cost/latency tradeoffs, that’s likely beyond GISA’s focus.

## Where GALILEO is weaker / needs to improve

- Consider adding (or aligning to) **structured outputs** where feasible to enable more deterministic scoring.
- Consider including/curating **gold trajectories** (or trajectory-quality evaluation) if process-level learning is a goal.
- Consider “live” / periodically refreshed evaluation slices to reduce contamination concerns.

## Action items for GALILEO (experiments / method / writing)

- [ ] In related work, cite GISA as a benchmark addressing backward-construction artifacts, contamination, and providing gold trajectories.
- [ ] Evaluate GALILEO (or a simplified variant) on GISA-style tasks: structured outputs + EM; report per-format breakdown.
- [ ] Consider adding a “live subset” (or time-split) evaluation protocol for GALILEO to argue robustness to memorization.
- [ ] If GALILEO trains with trajectories, compare to GISA’s human trajectories as a supervision source/baseline.

## Quotes / details to potentially cite

- “Existing benchmarks often construct queries backward from answers, producing unnatural tasks misaligned with real-world needs.”
- “GISA … comprising 373 human-crafted queries that reflect authentic information-seeking scenarios.”
- “GISA features four structured answer formats (item, set, list, and table), enabling deterministic evaluation.”
- “GISA provides complete human search trajectories for every query … for process-level supervision and imitation learning.”
- “Even the best-performing model achieves only 19.30% exact match score … degrading on tasks requiring complex planning and comprehensive information gathering.”
