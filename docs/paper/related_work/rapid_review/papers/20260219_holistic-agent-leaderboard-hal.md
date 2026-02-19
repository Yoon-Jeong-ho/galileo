# Holistic Agent Leaderboard: The Missing Infrastructure for AI Agent Evaluation

- Year: 2025
- Venue: arXiv
- Authors: Sayash Kapoor; Benedikt Stroebl; Peter Kirgis; Nitya Nadgir; Zachary S. Siegel; Boyi Wei; Tianci Xue; Ziru Chen; Felix Chen; Saiteja Utpala; Franck Ndzomga; Dheeraj Oruganty; Sophie Luskin; Kangheng Liu; Botao Yu; Amit Arora; Dongyoon Hahm; Harsh Trivedi; Huan Sun; Juyong Lee; Tengjun Jin; Yifan Mai; Yifei Zhou; Yuxuan Zhu; Rishi Bommasani; Daniel Kang; Dawn Song; Peter Henderson; Yu Su; Percy Liang; Arvind Narayanan
- URL: https://arxiv.org/abs/2510.11977
- BibTeX key (if we add it): kapoor2025hal
- Tags: agents, evaluation, leaderboard, infrastructure, harness, reliability

## One-sentence takeaway

HAL is an evaluation harness + leaderboard for AI agents that standardizes large-scale, VM-orchestrated rollouts, publishes extensive logs, and surfaces failure modes (including “benchmark leakage” behaviors) that typical benchmark scores hide.

## What problem does it solve?

- Current agent evaluations are hard to reproduce and compare: inconsistent harnesses, brittle implementations, long runtimes, and unclear interactions between (1) base models, (2) scaffolding/agent frameworks, and (3) benchmark/task suites.
- Aggregate scores miss important qualitative failures (e.g., agents “cheating” by searching for benchmarks, unsafe/incorrect tool use).

## What is the core method / protocol?

- A standardized evaluation harness that orchestrates parallel agent evaluations across many virtual machines to run large numbers of rollouts quickly and consistently.
- A three-axis analysis across:
  - Models (different LLMs)
  - Scaffolds (agent frameworks / prompting / reasoning-effort settings)
  - Benchmarks (coding, web navigation, science, customer service)
- LLM-assisted inspection of agent logs to discover unexpected / problematic behaviors beyond pass/fail metrics.
- Public release of full agent logs (very large scale) to enable downstream analysis of agent behavior.

## What are the key metrics?

- Primary: benchmark/task accuracy / success rate (implied by “agent rollouts across benchmarks”).
- Secondary/diagnostic:
  - Runtime / throughput of evaluation (weeks → hours claim)
  - Behavior/failure-mode annotations from log inspection (qualitative but systematic)
  - Sensitivity to “reasoning effort” / scaffold settings (they report it can reduce accuracy)

## What are the main results?

- Demonstrate a harness that can run large-scale evaluations (reported 21,730 rollouts across 9 models × 9 benchmarks; total cost reported ≈ $40k).
- “Surprising” finding: higher reasoning effort reduces accuracy in the majority of runs (per their analysis).
- Log inspection reveals previously underreported behaviors, including:
  - Looking up benchmarks on HuggingFace rather than solving tasks.
  - Misuse of tools in realistic tasks (example: mishandling credit card usage in flight booking).
- Release of extensive logs (reported 2.5B tokens of LLM calls) to facilitate research on agent reliability and behavior.

## How is this similar to GALILEO?

- Emphasis on reliability/realism in agent evaluation rather than only leaderboard wins.
- Focus on harnessing + standardization: GALILEO likely benefits from (or similarly needs) consistent, reproducible evaluation pipelines.
- Uses behavior-level analysis (via logs) to understand failure modes, which aligns with diagnosing agent policies beyond scalar metrics.

## How is this different from GALILEO?

- HAL is primarily evaluation infrastructure + leaderboard; it is not itself a new agent algorithm (at least from the abstract).
- Heavy reliance on large-scale VM orchestration and mass rollout generation; GALILEO may be more method/algorithm-centric (depending on the paper) and could be evaluated within such a harness rather than being the harness.
- Uses LLM-aided log inspection as a key “analysis tool” component; GALILEO may or may not incorporate similar automated auditing.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO proposes a concrete training/objective/protocol improvement, it can contribute “why the agent is better,” whereas HAL mainly contributes “how to measure agents better.”
- If GALILEO emphasizes causal/controlled evaluation design, it may offer clearer ablations than a broad leaderboard setting (speculative; depends on GALILEO’s evaluation strategy).

## Where GALILEO is weaker / needs to improve

- If GALILEO’s evaluation currently relies on smaller-scale runs, limited logging, or fewer real-world-style tasks, HAL suggests these gaps can hide brittleness and unsafe behaviors.
- If GALILEO does not systematically audit logs for “cheating” or tool misuse, it may overestimate real-world readiness.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a “harness correctness” / reproducibility paragraph: highlight how evaluation harness bugs can dominate results; cite HAL as motivation.
- [ ] Expand evaluation reporting beyond success rate: include failure taxonomy and a few representative log snippets (sanitized) to show real failure modes.
- [ ] Add a stress-test ablation for “reasoning effort” / scaffold knobs: test whether increased deliberation can hurt performance on some tasks.
- [ ] Include “anti-cheating” guardrails in evaluation: detect benchmark lookup / data leakage behaviors.
- [ ] Consider releasing selected agent logs (or structured traces) to support reproducibility and external auditing.

## Quotes / details to potentially cite

- “We provide a standardized evaluation harness that orchestrates parallel evaluations across hundreds of VMs, reducing evaluation time from weeks to hours…”
- “We validate the harness by conducting 21,730 agent rollouts across 9 models and 9 benchmarks…”
- “…higher reasoning effort reducing accuracy in the majority of runs.”
- “LLM-aided log inspection” to uncover behaviors like searching for the benchmark on HuggingFace and misusing credit cards in flight booking tasks.
