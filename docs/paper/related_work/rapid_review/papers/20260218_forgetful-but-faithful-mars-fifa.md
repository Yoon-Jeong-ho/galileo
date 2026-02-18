# Forgetful but Faithful: A Cognitive Memory Architecture and Benchmark for Privacy-Aware Generative Agents

- Year: 2025
- Venue: arXiv
- Authors: Saad Alqithami (per arXiv)
- URL: https://arxiv.org/abs/2512.12856
- BibTeX key (if we add it): alqithami2025forgetful
- Tags: agents, memory, forgetting, privacy, benchmark

## One-sentence takeaway

A cognitively inspired, typed memory store (MaRS) plus explicit forgetting policies and a multi-metric benchmark (FiFA) show that “forgetting-by-design” can improve long-horizon agent coherence while reducing cost and privacy leakage under fixed memory budgets.

## What problem does it solve?

- Long-running LLM agents accumulate unbounded memories; naive “store everything” is costly and raises privacy risk, while naive forgetting (e.g., windowing / random drops) breaks coherence and goal continuity.
- Lack of principled, budgeted retention policies and lack of evaluation suites that jointly measure task quality + social recall + privacy + cost under memory constraints.

## What is the core method / protocol?

- **MaRS (Memory-Aware Retention Schema):** a structured memory representation where memories are *typed* (episodic, semantic, social, task) and stored as nodes with metadata such as provenance, timestamps, token/weight cost, and sensitivity; retrieval is supported via multiple indices.
- **Forgetting policies (6):** FIFO, LRU, Priority Decay, Reflection-Summary, Random-Drop, and a **Hybrid** policy that stages/combines mechanisms to balance coherence vs efficiency vs privacy.
- **Privacy-aware retention:** the paper claims optional sensitivity-aware retention and the possibility of differential privacy guarantees ((ε,δ)-DP) in the retention process.
- **FiFA benchmark:** evaluates agents across narrative coherence, goal completion, social recall accuracy, privacy preservation, and cost efficiency across multiple **memory budgets** (reported: 5 budgets) and configurations (reported: 300 runs).

## What are the key metrics?

- Narrative coherence (long-horizon consistency)
- Goal completion (multi-step task success)
- Social recall accuracy (remembering people/relationships correctly)
- Privacy preservation / leakage
- Cost efficiency / computational tractability (under token/memory budgets)
- Composite score aggregating the above (paper reports a composite ≈ 0.911 for best method)

## What are the main results?

- Across 300 runs and varying memory budgets, the **Hybrid** forgetting policy reportedly achieves the best overall composite performance (≈0.911), while keeping cost tractable and privacy scores high.
- Baselines that are purely temporal (FIFO/LRU) or simplistic (random drop) tend to trade off coherence and recall for efficiency; reflection/summary helps but still requires principled eviction decisions.

## How is this similar to GALILEO?

- Both care about **long-horizon agent behavior** where memory management is a first-order design choice.
- Emphasizes **human-centered evaluation axes** beyond raw task success (e.g., coherence, social consistency).
- Treats memory as structured (not just a raw transcript), aligning with architectures that separate memory types and apply policy to retention/retrieval.

## How is this different from GALILEO?

- This work is primarily about **memory budgeting + forgetting policies + benchmark design**, rather than GALILEO’s core technical focus (as currently framed) on the specific GALILEO methodology/setting.
- Introduces an explicit benchmark suite (FiFA) with privacy and cost metrics baked in; GALILEO’s evaluation may not yet foreground privacy leakage and “right-to-forget” style constraints.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO has tighter problem definitions, stronger empirical baselines, or clearer causal ablations, it may provide a cleaner demonstration than a broad architecture+benchmark proposal.
- If GALILEO avoids hand-designed policy heuristics (or makes them fully explicit/ablated), it can look more reproducible.

## Where GALILEO is weaker / needs to improve

- Add explicit **memory budget** sweeps and show quality/cost curves.
- Add **privacy-aware** metrics (or at least a leakage proxy) and show that GALILEO does not over-retain sensitive details.
- Provide clearer taxonomy of memory types (episodic/semantic/social/task) and show which are essential for GALILEO’s success.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding a “forgetting-by-design” paragraph in related work, citing MaRS/FiFA as a representative benchmark+policy framework for budgeted memory.
- [ ] Add an experiment: fixed memory budget vs performance, with at least FIFO/LRU vs an importance-aware policy (priority decay / summary) as baselines.
- [ ] If relevant, include a privacy section: how GALILEO treats sensitive personal info in memory; propose a simple sensitivity tag + retention rule.

## Quotes / details to potentially cite

- “We present the Memory‑Aware Retention Schema (MaRS) … [with] episodic, semantic, social, and task memories … typed, provenance‑tracked nodes …” (arXiv HTML abstract).
- “We introduce the FiFA benchmark … [measuring] narrative coherence, goal completion, social recall, privacy preservation, and cost efficiency.” (arXiv HTML abstract).
- “Across 300 simulation runs spanning five memory budgets, the Hybrid policy delivers the best composite performance (≈0.911) …” (arXiv HTML abstract).
