# ProAgentBench: Evaluating LLM Agents for Proactive Assistance with Real-World Data

- Year: 2026
- Venue: arXiv
- Authors: Huaze Tang; Tingyu Cao; Lam Nguyen; Anping Zhang; Xinwen Cao; Chunkang Liu; Wenbo Ding; Yang Li
- URL: https://arxiv.org/html/2602.04482v1
- BibTeX key (if we add it): ProAgentBench2026Tang
- Tags: agents, proactive-assistance, benchmark, real-world-logs, screenshots, long-term-context

## One-sentence takeaway

A benchmark + real-world (privacy-filtered) dataset for proactive agents that decomposes “proactivity” into **when to interrupt** and **what to say**, showing large gains from longer history and explicit long-term memory.

## What problem does it solve?

- Existing “proactive agent” datasets are either (i) heavily LLM-synthesized (missing authentic temporal patterns) or (ii) short/isolated tasks (missing **pre-assistance context** that could signal a need for help).
- Evaluations often conflate *timing* (should the agent intervene now?) with *content* (what assistance should it provide?), making it hard to measure tradeoffs like alert fatigue vs coverage.

## What is the core method / protocol?

- Collect continuous real user desktop sessions (screenshots at ~1Hz + app/window metadata), segment into events by app switching, and identify LLM-related events.
- Privacy pipeline (as described): VLM pre-screening + human-in-the-loop review + rule-based filtering; high-risk data deleted.
- Define a hierarchical proactive-assistance evaluation:
  - **When to Assist**: binary trigger prediction given user meta-info + recent observation history.
  - **How to Assist**: conditional generation of assistance content when the trigger fires.
- Evaluate prompt-based baselines (zero-shot / CoT / self-consistency) and memory-based variants (RAG / knowledge graph / clustering) on both tasks.

## What are the key metrics?

- When to Assist: Accuracy, Precision, Recall, F1 (explicitly tied to interruption costs vs missed-help costs).
- How to Assist: (i) Intention Accuracy (coarse intent class), (ii) Semantic Similarity (embedding cosine vs actual user query).
- Dataset characterization: burstiness score B (Goh & Barabasi) + heavy-tail IET fit comparisons (power-law vs exponential).

## What are the main results?

- Real user logs exhibit strong burstiness (reported B≈0.787) and heavy-tailed inter-event times; LLM-synthesized interactions (even with “realistic candidate time points”) look much less bursty (reported B≈0.166).
- Longer short-term history helps; authors highlight diminishing returns beyond about a 5-minute window for some intention metrics.
- Adding long-term memory helps substantially; their best reported memory approach is a **knowledge-graph** style memory organization (largest gains over zero-shot among their tested options).
- Fine-tuning on real-world data beats fine-tuning on synthetic data for open models (large deltas reported in accuracy / intention accuracy).

## How is this similar to GALILEO?

- Shares the core framing that **context over time** (short-term trajectory + longer-term memory) is critical to measuring/understanding agent behavior.
- Emphasizes evaluation beyond single-turn snapshots, and explicitly argues that naive prompting strategies (e.g., CoT) can be counterproductive in certain settings.

## How is this different from GALILEO?

- Focuses on **human-computer proactive assistance** (interruption timing + helpful suggestion generation) from screen-activity logs, rather than conversational belief-drift / pressure / multi-turn robustness.
- Heavy dependence on a privacy-filtered screenshot-log dataset and VLM annotation pipeline; GALILEO’s core contributions are not primarily dataset collection.

## Where GALILEO is stronger / cleaner (if true)

- GALILEO can likely provide cleaner causal controls for *why* behavior changes across turns (e.g., pressure-only vs evidence-driven change), whereas ProAgentBench is more about ecological realism and task decomposition.

## Where GALILEO is weaker / needs to improve

- If GALILEO claims to cover “agents in the wild”, this paper is a reminder that real interaction streams are **bursty** and long-horizon; we may need to justify how our protocol approximates such temporal structure.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding a short “why synthetic interaction timing is unrealistic” note: burstiness / heavy-tail IET is an intuitive, reportable statistic.
- [ ] If we discuss memory/context length, cite their ablation-style result that longer recent history improves prediction, with diminishing returns after a few minutes.
- [ ] Borrow the explicit precision/recall interpretation as **alert fatigue vs coverage** when we discuss any trigger/monitoring sub-component.

## Quotes / details to potentially cite

- Abstract claim: “28,000+ events from 500+ hours of real user sessions” and “burstiness B=0.787”.
- Problem decomposition: “When + How” hierarchical framework (timing prediction + assist content generation).
- Reported finding: real-world training data outperforms synthetic alternatives; memory/historical context improves prediction accuracy.
