# CM2: Reinforcement Learning with Checklist Rewards for Multi-Turn and Multi-Step Agentic Tool Use

- Year: 2026
- Venue: arXiv
- Authors: Zhen Zhang; Kaiqiang Song; Xun Wang; Yebowen Hu; Weixiang Yan; Chenyang Zhao; Henry Peng Zou; Haoyun Deng; Sathish Reddy Indurthi; Shujian Liu; Simin Ma; Xiaoyang Wang; Xin Eric Wang; Song Wang
- URL: https://arxiv.org/abs/2602.12268
- BibTeX key (if we add it): cm2_zhang_2026
- Tags: tool-use, multi-turn, RL, agents, reward-design

## One-sentence takeaway

CM2 proposes an RL recipe for multi-turn tool-using agents that replaces hard-to-verify outcome rewards with **checklist-style binary criteria** (with explicit evidence grounding), enabling scalable RL improvements over SFT in simulated tool environments.

## What problem does it solve?

- RL for real tool-using agents is hard because many objectives are **not verifiable** (open-ended “did the agent behave correctly?”), multi-turn/multi-step tool use is underexplored, and building executable tool environments is expensive.
- Prior RL setups often depend on **verifiable end rewards** (success/failure) that don’t exist for many assistant tasks.

## What is the core method / protocol?

- Replace outcome reward with **checklist rewards**:
  - Decompose a turn’s desired behavior into fine-grained **binary criteria**.
  - Each criterion is judged with **explicit evidence grounding** + structured metadata, turning “open-ended judging” into more stable classification-like decisions.
- Training in an **LLM-simulated tool environment** to avoid engineering many real tool backends.
- Design choice: **sparse reward assignment** (don’t reward constantly) but **dense evaluation criteria** (many checks when you do evaluate) to balance stability vs informativeness.

## What are the key metrics?

- Reported deltas vs supervised fine-tuning (SFT) on tool-agent benchmarks:
  - tau^-Bench
  - BFCL-V4
  - ToolSandbox

## What are the main results?

- With an 8B base model + 8k-example RL dataset, CM2 improves over SFT by:
  - +8 on tau^-Bench
  - +10 on BFCL-V4
  - +12 on ToolSandbox
- They claim results are competitive with similarly sized open-source baselines, including the judging model.

## How is this similar to GALILEO?

- Both are concerned with **multi-turn trajectories** where single scalar outcomes are inadequate.
- The “dense criteria / structured evaluation” idea resonates with how we might want to score **trajectory-level failure modes** (drift, flip, recovery) with more than one label.

## How is this different from GALILEO?

- CM2 is primarily a **training method (RL reward design)** for tool-using agents, not an evaluation framework for social-pressure drift / belief revision.
- Their primary challenge is **reward construction** and scalable environments; our core focus is **pressure-vs-evidence dynamics**, time-to-failure, and recovery/oscillation structure.

## Where GALILEO is stronger / cleaner (if true)

- If our protocols separate **pressure-only** from **evidence-bearing** updates, we have a clearer causal interpretation than generic checklist-based “good behavior” scoring.
- If our evaluation uses real interaction settings (or well-controlled perturbations) rather than purely simulated tools, we may better capture deployment failure modes.

## Where GALILEO is weaker / needs to improve

- We may lack an explicit recipe for **training** multi-turn agents against our failure modes (beyond evaluation + prompting baselines).
- Our scoring might be too coarse if we rely on a small set of trajectory metrics; CM2 suggests a path toward **more granular, evidence-grounded criteria**.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider a “checklist rubric” for our trajectory annotations (e.g., per turn: evidence introduced? evidence used? belief flip? justification quality? recovery attempt?) to improve judge stability.
- [ ] If we ever do RL-style mitigation, test whether **multi-criteria binary rubrics** outperform single outcome rewards for reducing pressure-driven drift while preserving receptiveness.

## Quotes / details to potentially cite

- “We propose CM2, an RL framework that replaces verifiable outcome rewards with checklist rewards.”
- “CM2 decomposes each turn's intended behavior into fine-grained binary criteria with explicit evidence grounding and structured metadata, turning open-ended judging into more stable classification-style decisions.”
- “Experiments show that CM2 consistently improves over supervised fine-tuning… improves over the SFT counterpart by 8 points on tau^-Bench, by 10 points on BFCL-V4, and by 12 points on ToolSandbox.”
