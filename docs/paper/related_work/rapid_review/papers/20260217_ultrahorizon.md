# UltraHorizon: Benchmarking Agent Capabilities in Ultra Long-Horizon Scenarios

- Year: 2025
- Venue: arXiv
- Authors: Haotian Luo, Huaisong Zhang, Xuelin Zhang, Haoyu Wang, Zeyu Qin, Wenjie Lu, Guozheng Ma, Haiying He, Yingsha Xie, Qiyang Zhou, Zixuan Hu, Hongze Mi, Yibo Wang, Naiqiang Tan, Hong Chen, Yi R. Fung, Chun Yuan, Li Shen
- URL: https://arxiv.org/abs/2509.21766
- BibTeX key (if we add it): ultrahorizonLuo2025
- Tags: agents, long-horizon, benchmark, robustness, degradation, tool-use, memory

## One-sentence takeaway

UltraHorizon proposes exploration-style long-horizon, partially-observable environments (tens-to-hundreds of thousands of tokens and many tool calls) that expose systematic capability gaps and “locking” failure modes in LLM agents versus humans.

## What problem does it solve?

- Current agent benchmarks are mostly short-horizon and/or fully observable, missing key real-world difficulties where agents must sustain planning, reasoning, memory, and tool use over long trajectories.
- Provides a stress-test benchmark explicitly targeting *ultra long-horizon* behavior with partial observability and iterative discovery.

## What is the core method / protocol?

- Benchmark = “exploration as a unifying task” across **three distinct environments**.
- Agents must iteratively interact with an environment to **discover hidden rules** (i.e., long-horizon discovery), requiring:
  - sustained reasoning + planning,
  - memory management (tracking hypotheses / observations),
  - tool-use management (many tool calls),
  - robustness to long-context accumulation.
- Reports scale settings where trajectories average **35k+ tokens with 60+ tool calls**, and heavy settings averaging **200k+ tokens with 400+ tool calls**.
- Provides qualitative/error analysis over collected trajectories; identifies **8 error types**, attributed to:
  - *in-context locking* (getting stuck in an unproductive pattern / hypothesis),
  - *fundamental capability gaps* (tool use, planning, memory, etc.).

## What are the key metrics?

- Benchmark scores per environment/task (details not fully visible from the abstract page alone).
- Comparative performance of LLM agents vs human participants.
- Token / tool-call scale statistics (trajectory length, number of tool calls).
- Error taxonomy counts/analysis from trajectories.

## What are the main results?

- LLM agents **consistently underperform** humans in long-horizon partially-observable settings.
- “Simple scaling” (implied: bigger models / more tokens) **does not** close the gap on this benchmark.
- Failure analysis suggests recurring lock-in behaviors and missing core competencies.

## How is this similar to GALILEO?

- Shared theme: evaluating *multi-step / multi-turn robustness* rather than single-turn accuracy.
- Highlights the importance of **trajectory-level** evaluation and diagnosing **failure modes over time**.
- Emphasizes memory and tool-use management as sources of degradation, which aligns with long-run instability concerns.

## How is this different from GALILEO?

- UltraHorizon targets **task completion in interactive exploration environments**, not (primarily) truthfulness, sycophancy, or being misled by adversarial dialogue.
- Focus is on **agent capability** under long-horizon partial observability, rather than *belief/answer stability* under pressure or adversarial conversational perturbations.
- Metrics appear to be environment/task success measures rather than survival/time-to-failure against explicit attacks.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO’s protocol is simpler to reproduce (fixed datasets + standardized perturbations), it may yield clearer attribution than complex interactive environments.
- GALILEO can focus directly on *robustness to misleading context / pressure*, which is a more targeted neighbor to sycophancy/drift phenomena.

## Where GALILEO is weaker / needs to improve

- Long-horizon token/tool-call regimes (100k+ tokens, hundreds of tool calls) may reveal failure modes that short/medium-horizon GALILEO settings miss.
- If GALILEO does not include partial observability + hypothesis tracking, it may under-cover real agentic settings.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider an “ultra long-horizon” variant of the protocol (or a stress-test track) with:
  - explicit memory bookkeeping requirements,
  - many-turn accumulation (e.g., 10^4–10^5 token contexts),
  - tool-use logging and tool-call budget.
- [ ] Add a brief related-work paragraph positioning: *agent benchmarks focus on long-horizon task success (e.g., UltraHorizon), while GALILEO targets long-horizon robustness/stability under misleading pressure with clearer causal controls.*
- [ ] Steal the idea of *trajectory error taxonomy* (define a small set of “drift/lock-in” error categories for GALILEO runs).

## Quotes / details to potentially cite

- “Under the heaviest scale setting, trajectories average **200k+ tokens** and **400+ tool calls**, whereas in standard configurations they still exceed **35k** tokens and involve more than **60** tool calls on average.”
- “We identify eight types of errors and attribute them to two primary causes: **in-context locking** and **functional fundamental capability gaps**.”
