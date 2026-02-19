# ProAgentBench: Evaluating LLM Agents for Proactive Assistance with Real-World Data

- Year: 2026
- Venue: arXiv
- Authors: Huaze Tang; Tingyu Cao; Lam Nguyen; Anping Zhang; Xinwen Cao; Chunkang Liu; Wenbo Ding; Yang Li
- URL: https://arxiv.org/abs/2602.04482
- BibTeX key (if we add it): tang2026proagentbench
- Tags: agents, proactive-assistance, evaluation, benchmark, real-world-logs, long-term-context, hci

## One-sentence takeaway

ProAgentBench proposes a “When + How” evaluation framework and a privacy-compliant real-user workflow dataset (28k+ events, 500+ hours) to benchmark proactive agents on timing prediction and assistance generation.

## What problem does it solve?

- Prior “proactive agent” datasets/benchmarks are often (a) LLM-synthesized (missing authentic human patterns) and/or (b) focus on isolated tasks, lacking the *pre-assistance* context needed to learn *when* to proactively intervene.
- There is also a privacy barrier to collecting realistic screen/workflow logs at scale.

## What is the core method / protocol?

- Benchmark + dataset for proactive assistance in working scenarios.
- Task decomposition (hierarchical):
  - **When to Assist**: predict optimal intervention timing (cast as a binary classification problem).
  - **How to Assist**: generate the assistance content given context.
- Data collection pipeline:
  - Privacy-compliant workflow logging (described as rule-based anonymization + human-in-the-loop review).
  - Captures continuous user work sessions to preserve temporal structure and pre-assistance context.
- Baselines: “LLM- and VLM-based” models evaluated; paper emphasizes value of long-term memory + historical context.

## What are the key metrics?

- For timing (“When to Assist”): standard classification metrics; the intro highlights **precision** (interruption cost / alert fatigue) and **recall** (coverage of needs).
- For retrieval of historical context: they reference similarity search over screenshots (embedding-based) and analyze time-to-event distributions (from the HTML).

## What are the main results?

- Using **long-term memory / historical context** improves timing prediction accuracy.
- Training/evaluating on **real-world data** substantially outperforms synthetic alternatives.
- Dataset scale/statistics highlighted:
  - 28,000+ events
  - 500+ hours of real user sessions
  - Bursty temporal patterns quantified (burstiness **B = 0.787**)

## How is this similar to GALILEO?

- If GALILEO targets proactive help in workflows, this is directly aligned: it frames proactive assistance as an always-on agent that decides **when** to interrupt and **what** to do.
- Strong emphasis on **long-term context** as a first-class input for decision making (timing + content).
- Provides benchmark framing that could be used to evaluate GALILEO-style systems.

## How is this different from GALILEO?

- This work is primarily a **benchmark/dataset + baseline evaluation** contribution, not a new end-to-end agent architecture.
- Focuses on **screen/workflow logging** and privacy-compliant collection; GALILEO may not rely on (or have access to) comparable real user screen datasets.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO has a clearer agentic protocol / tool-use spec (beyond “when/how”), it may provide more actionable system design guidance than a benchmark-focused paper.

## Where GALILEO is weaker / needs to improve

- If GALILEO lacks an evaluation story for proactive intervention, ProAgentBench’s explicit **timing-vs-content** decomposition and precision/recall framing is a useful reference point.
- If GALILEO evaluation relies on synthetic logs/tasks, this paper is a reminder that synthetic data can miss burstiness and other real-work dynamics.

## Action items for GALILEO (experiments / method / writing)

- [ ] In related work, cite the **“When + How”** decomposition and map GALILEO components onto it.
- [ ] Consider adding a timing-eval section using **precision/recall** with an explicit “interruption cost” interpretation.
- [ ] If GALILEO uses memory/context, highlight that this paper finds **historical context + long-term memory** materially improve prediction.
- [ ] Add a short discussion on **bursty** user interaction patterns and why continuous-session data matters.

## Quotes / details to potentially cite

- “Our dataset captures over **28,000 events** from **500+ hours** of continuous working sessions…”
- “…preserving the bursty temporal patterns (**burstiness B = 0.787**) that synthetic data fundamentally lacks.”
- “We then formalize a ‘**When + How**’ hierarchical task framework that decomposes proactive assistance into… **When to Assist** … and **How to Assist** …”
- Precision/recall interpretation (intro): precision as interruption cost / alert fatigue; recall as need coverage / workflow fragmentation.
