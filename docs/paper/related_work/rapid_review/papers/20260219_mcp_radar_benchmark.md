# MCP-Radar: A Multi-Dimensional Benchmark for Evaluating Tool Use Capabilities in Large Language Models

- Year: 2025
- Venue: arXiv (cs.AI)
- Authors: Xuanqi Gao; Siyi Xie; Juan Zhai; Shiqing Ma; Chao Shen
- URL: https://arxiv.org/abs/2505.16700
- BibTeX key (if we add it): mcp-radar-2025-gao
- Tags: tool-use, mcp, benchmark, evaluation, agent, efficiency

## One-sentence takeaway

MCP-Radar introduces a 507-task, MCP-native benchmark with objective scoring that separates “got the answer” from “did the right tool operations,” and surfaces an accuracy–efficiency trade-off plus common tool-selection failures.

## What problem does it solve?

- MCP is becoming a de-facto standard for tool discovery/orchestration, but existing tool-use evals either (a) don’t really test MCP-style interactions, (b) rely on subjective/binary judgments, or (c) use low-fidelity simulations that miss real tool interaction complexity.
- Prior benchmarks also struggle to distinguish genuine tool-based problem solving from models “reciting” memorized facts.

## What is the core method / protocol?

- Build an evaluation suite specifically around the MCP paradigm.
- Dataset: 507 tasks across 6 domains:
  - mathematical reasoning
  - web search
  - email
  - calendar
  - file management
  - terminal operations
- Two task families:
  - **Precise Answer** tasks (single ground-truth value; e.g., Math/Websearch)
  - **Fuzzy Match** tasks (need correct *sequence* of operations; e.g., File/Terminal)
- Tooling setup:
  - Mix of **real MCP tools** (e.g., from tool platforms) + **high-fidelity mock MCP tools** (email/calendar etc.) aligned with official specs.
- Evaluation mechanics:
  - **Answer Matching** for Precise Answer
  - **Operation Matching** for Fuzzy Match

## What are the key metrics?

- Two primary correctness notions:
  - **Answer correctness**
  - **Operational accuracy** (whether the tool interaction sequence is correct)
- Reported multi-dimensional metrics (naming as in the paper):
  - **Answer Accuracy (RA)**
  - **Tool Selection Efficiency (DTSR)**
  - **Computational Resource Efficiency (CRE)**
- Additional quantified aspects mentioned: number of successful tool-invocation rounds (and other efficiency/resource proxies).

## What are the main results?

- Capability profiles differ substantially across models and domains.
- Closed-source models lead strongly on math; on web search the closed/open gap narrows (paper claims to <10%).
- A recurring failure mode: selecting a **semantically plausible but functionally wrong tool**, suggesting shallow task/tool understanding.
- Clear **accuracy vs efficiency** trade-off: some models do better but use more tool rounds / compute.

## How is this similar to GALILEO?

- Directly relevant if GALILEO is an LLM agent that uses tools: MCP-Radar’s split between *answer correctness* and *operation correctness* matches what we care about for reliable agents.
- The “wrong tool chosen” failure mode is a common agent issue; the benchmark provides a structured way to measure it.

## How is this different from GALILEO?

- MCP-Radar is an **evaluation benchmark + metrics framework**, not a new agent architecture.
- Emphasis is on MCP-standardized tool orchestration and objective measurement, rather than task-specific prompting/agent policy.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO focuses on end-to-end reliability (planning, state tracking, recovery), it may provide stronger *method-side* contributions than a benchmark alone.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently reports only task success rates, it may be missing MCP-Radar-style **operational accuracy** and **efficiency** dimensions.
- If GALILEO uses ad-hoc tool APIs, MCP compatibility/claims may be weaker without an MCP-native evaluation.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add MCP-Radar to evaluation (or at least mirror its task taxonomy: Precise Answer vs Fuzzy Match).
- [ ] In the paper, explicitly separate metrics: answer correctness vs operational/tool-trajectory correctness.
- [ ] Report an efficiency metric (tool-call rounds, latency proxy, token/tool budget) alongside accuracy.
- [ ] Add an ablation/analysis section for “tool selection errors” (semantically plausible but wrong tool) and how GALILEO mitigates them.

## Quotes / details to potentially cite

- “MCP-Radar features a challenging dataset of 507 tasks spanning six domains: mathematical reasoning, web search, email, calendar, file management, and terminal operations.” (arXiv abstract)
- “It quantifies performance based on two primary criteria: answer correctness and operational accuracy.” (arXiv abstract)
- Introduces objective metrics beyond binary success, including efficiency and successful tool-invocation rounds; and highlights an accuracy–efficiency trade-off plus tool-misselection failures (Introduction/Abstract).
