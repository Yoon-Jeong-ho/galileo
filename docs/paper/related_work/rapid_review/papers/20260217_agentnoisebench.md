# AgentNoiseBench: Benchmarking Robustness of Tool-Using LLM Agents Under Noisy Condition

- Year: 2026
- Venue: (listed as) ICML (arXiv preprint)
- Authors: Yuxin Chen; Yukai Wang; Chang Wu; Junfeng Fang; Xiaodong Cai; Qi Gu; Hui Su; An Zhang; Xiang Wang; Xunliang Cai; Tat-Seng Chua
- URL: https://arxiv.org/abs/2602.11348
- BibTeX key (if we add it): agentnoisebench2026chen
- Tags: agents, tool-use, robustness, noise, evaluation, multi-turn

## One-sentence takeaway

AgentNoiseBench evaluates tool-using/search agents under *realistic user/tool noise* via solvability-preserving perturbations and shows current agents degrade substantially—especially under tool failures—highlighting a robustness gap not captured by “clean” benchmarks.

## What problem does it solve?

- Standard agent benchmarks assume idealized conditions (clean instructions; stable tools), which overestimates real-world performance.
- There is no unified way to (i) categorize realistic noise sources, (ii) inject them without breaking task solvability, and (iii) evaluate *trajectory-level* robustness.

## What is the core method / protocol?

- Proposes a framework with three pillars:
  - **Noise taxonomy**: two high-level sources
    - **User-noise**: ambiguity/variability in user instructions and interaction patterns.
    - **Tool-noise**: failures/inconsistencies/partial outputs from external tools.
  - **Solvability-preserving noise injection**: an automated pipeline that injects controllable perturbations into existing agent benchmarks while keeping tasks solvable (so failures are attributable to agent brittleness, not broken tasks).
  - **Trajectory-aware evaluation**: emphasizes that outcome-only success can be “lucky”; evaluates robustness by analyzing intermediate steps / procedural integrity under noise.
- Instantiates the benchmark by injecting noise into multiple existing agent settings (tool-use + search; includes a multi-hop QA setting).

## What are the key metrics?

- Primary: benchmark task scores under different noise regimes (clean vs user-noise vs tool-noise; potentially with varying noise strengths).
- Trajectory-aware analyses mentioned:
  - step-wise entropy trends along trajectories (used to diagnose how noise changes “entropy minimization” dynamics)
  - failure pattern analysis (which noise source causes which failures; sensitivity by model family)

## What are the main results?

- **Consistent performance degradation** across a broad range of open/proprietary models when noise is introduced.
- **Tool-noise tends to hurt more than user-noise** (models are broadly more sensitive to unreliable tool outputs).
- **General reasoning ability ≠ robustness**: stronger “reasoning” models are not necessarily more robust to environmental perturbations.

## How is this similar to GALILEO?

- Same high-level motivation: single-number “clean” performance misses *interaction dynamics*; robustness should be evaluated under realistic perturbations.
- Emphasizes multi-turn / trajectory structure and environment-induced degradation (not just static QA accuracy).

## How is this different from GALILEO?

- Focuses on **agent-environment noise** (instruction ambiguity + tool unreliability), not primarily on **social pressure / persuasion / sycophancy**.
- Evaluation centers on task success + procedural robustness under noise, rather than explicit *belief drift vs evidence-driven revision* controls.

## Where GALILEO is stronger / cleaner (if true)

- GALILEO’s core positioning (pressure-driven drift vs evidence-driven revision, recovery dynamics, time-to-failure framing) targets a different causal mechanism than generic noise.
- GALILEO can likely provide cleaner *paired* controls (pressure vs evidence) than generic noisy-environment perturbations.

## Where GALILEO is weaker / needs to improve

- If GALILEO claims relevance to agentic deployments, this paper is a reminder that **tool unreliability is a major driver of multi-turn failures**; GALILEO may need at least one agentic/tool slice to avoid being “chat-only”.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a short related-work paragraph: “agent robustness under noisy environments” as a complementary axis to persuasion/pressure robustness.
- [ ] Consider an optional agentic ablation: evaluate drift/flip metrics when tools intermittently fail / return partial results (to show whether GALILEO’s metrics generalize beyond dialogue-only perturbations).
- [ ] If we add an agent slice, report separate sensitivity to **user-noise** vs **tool-noise** (mirrors their taxonomy).

## Quotes / details to potentially cite

- “We categorize environmental noise into two primary types: user-noise and tool-noise.”
- “Injects controllable noise into existing agent-centric benchmarks while preserving task solvability.”
- “Outcome-based metrics are insufficient … trajectory-aware evaluation … procedural integrity.”
- “Models … are more sensitive to tool-noise than to user-noise.”
