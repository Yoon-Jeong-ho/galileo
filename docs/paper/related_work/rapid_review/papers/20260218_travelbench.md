# TravelBench: A Broader Real-World Benchmark for Multi-Turn and Tool-Using Travel Planning

- Year: 2026
- Venue: arXiv
- Authors: Xiang Cheng, Yulan Hu, Xiangwen Zhang, Lu Xu, Zheng Pan, Xin Li, Yong Liu
- URL: https://arxiv.org/abs/2512.22673
- BibTeX key (if we add it): TravelBench2026
- Tags: multi-turn, tool-use, planning, benchmark, robustness, unsolvable-detection

## One-sentence takeaway

TravelBench is a real-world travel-planning benchmark (single-turn, multi-turn, and unsolvable) with cached tool outputs in a sandbox to make tool-using agent evaluation reproducible and to test boundary recognition.

## What problem does it solve?

- Existing travel-planning agent benchmarks are less “real-world” due to limited domain/task coverage, simplistic or explicitly stated preferences (vs implicit preferences revealed through dialogue), and weak evaluation of an agent’s capability limits (e.g., whether it should ask clarifying questions or admit infeasibility).
- Tool-using evaluation often suffers from non-reproducible external tool results.

## What is the core method / protocol?

- Dataset + evaluation environment design:
  - Collects real user travel queries plus user profiles/preferences and a set of travel-related tools.
  - Defines 3 subtasks to probe distinct capabilities:
    1) **Single-Turn**: solve autonomously from one query.
    2) **Multi-Turn**: interact over multiple turns to elicit missing/implicit requirements.
    3) **Unsolvable**: detect/acknowledge infeasible requests (capability boundary recognition).
- Reproducibility infrastructure:
  - Integrates **10 travel tools** (e.g., POI/flight/train search, route planning, weather) into a sandbox.
  - **Caches tool-call results** (reported ~200k tool responses) so tool-augmented runs are stable and comparable.
- Scoring:
  - Uses an **LLM-as-a-judge** protocol with a unified rubric to score response quality / task completion.

## What are the key metrics?

- Primary: LLM-judge scores for task completion / response quality under a unified rubric (details not fully extracted in this rapid pass).
- Implicitly: success rates across the three subsets (single-turn vs multi-turn vs unsolvable) and behavioral analyses of tool use / interaction failures.

## What are the main results?

- Introduces a larger-scope, more realistic benchmark with:
  - **1,100 instances** (reported: 500 single-turn, 500 multi-turn, 100 unsolvable).
  - A tool cache and sandbox enabling reproducible tool-use evaluation.
- Empirical evaluation across multiple LLMs with analysis of behaviors/performance (numbers not captured from the truncated HTML in this rapid pass).

## How is this similar to GALILEO?

- Shared evaluation themes:
  - Multi-turn interaction as a first-class axis (not just single-shot answers).
  - Explicit focus on **capability boundaries** (their “Unsolvable” split) — i.e., when a system should refuse / ask for clarification / admit it cannot complete the task.
  - Concern for **robustness and reproducibility** of evaluations.

## How is this different from GALILEO?

- Domain-specific: travel planning with a concrete toolbox (POI/route/weather/transport) rather than (presumably) more general conversational robustness settings.
- Heavy emphasis on **tool-use sandbox + caching** as the central reproducibility mechanism.
- Uses LLM-as-a-judge scoring as the main evaluation mechanism (may differ from GALILEO’s metrics/protocol).

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO uses more direct, mechanically-checkable metrics (vs judge-based), it may offer less subjective evaluation.
- If GALILEO targets broader failure modes beyond a single application domain, it may generalize better.

## Where GALILEO is weaker / needs to improve

- If GALILEO evaluates tool use, TravelBench’s **tool-call caching + sandbox** is a strong pattern for reproducibility.
- If GALILEO lacks an explicit “unsolvable / admit limits” slice, TravelBench provides a clean benchmark design template.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding an explicit **unsolvable / infeasible** subset (or labeling) to measure boundary recognition separately from task success.
- [ ] If tool calls are part of the evaluation, consider **caching tool outputs** and running in a sandboxed harness to make comparisons reproducible.
- [ ] Add a “multi-turn preference elicitation” axis: measure whether the agent can discover implicit constraints via questions rather than relying on the user to specify everything.

## Quotes / details to potentially cite

- “We cache real tool-call results and build a sandbox environment that integrates ten travel-related tools.”
- Subtasks: “Single-Turn, Multi-Turn, and Unsolvable” to evaluate autonomy, multi-turn clarification, and boundary recognition.
- Scale details from the paper HTML: 500 single-turn, 500 multi-turn, 100 unsolvable; tool-call cache ~200,000 responses; 10 tools.
