# Stabilizing Long-term Multi-turn Reinforcement Learning with Gated Rewards

- Year: 2025
- Venue: arXiv
- Authors: Zetian Sun; Dongfang Li; Zhuoen Chen; Yuhuai Qin; Baotian Hu
- URL: https://arxiv.org/abs/2508.10548
- BibTeX key (if we add it): sun2025gated
- Tags: RL, long-horizon, multi-turn, reward-shaping, SWE-agents, verification-based-rewards

## One-sentence takeaway

They propose **Gated Reward Accumulation (G-RA)**: only “unlock” dense stepwise/format/tool-use rewards when the sparse long-term objective clears a threshold, preventing reward hacking/policy collapse in multi-turn SWE RL.

## What problem does it solve?

- In long-horizon / multi-turn RL (esp. SWE agent settings), sparse outcome rewards make learning hard.
- Adding dense, verifiable stepwise rewards (format/tool-use/heuristics) can *destabilize* training because those rewards are easier to optimize than the true objective.
- This misalignment can cause reward hacking and even catastrophic policy degradation (e.g., optimizing easy stepwise rewards while never completing the task).

## What is the core method / protocol?

- **SWE-oriented RL Framework**: multi-turn LM ↔ environment interaction with a 4-layer architecture (policy → scaffold → low-level interface → environment), docker-based execution, customizable scaffolds/rewards.
- **Gated Reward Accumulation (G-RA)**:
  - Organize rewards by priority: high-level long-term outcome reward vs low-level stepwise critics.
  - If the high-level reward is below a threshold, **mask / zero out** lower-priority stepwise rewards.
  - SWE instantiation: if the model fails to submit a patch or submits an empty patch, stepwise rewards are not counted.

## What are the key metrics?

- **CR (Completion Rate)**: proportion of trajectories where the agent calls the submit/finish action.
- **MR (Modification Rate)**: proportion of trajectories where the repository is modified before submission.
- **RR (Resolution Rate)**: proportion of tasks successfully solved (pass tests).

## What are the main results?

- On **SWE-bench Verified** and **kBench-50**, G-RA substantially improves long-horizon behavior vs direct reward accumulation (D-RA), per abstract:
  - SWE-bench Verified: CR 47.6% → 93.8%, MR 19.6% → 23.8%
  - kBench: CR 22.0% → 86.0%, MR 12.0% → 42.0%
- D-RA shows a characteristic failure mode: total reward increases while outcome reward decreases (classic reward hacking / misalignment).
- Gate-threshold ablations: too-low thresholds allow collapse; too-high thresholds underuse useful stepwise signals.

## How is this similar to GALILEO?

- Same core concern: **multi-turn / long-horizon optimization** where intermediate signals (heuristics, verification checks) can dominate the true objective.
- Their analysis/diagnosis (reward hacking, collapse, misaligned dense vs sparse signals) is closely aligned with robustness framing: “what you measure/optimize” can diverge from “what you want.”

## How is this different from GALILEO?

- Focus is **training-time RL reward design** for SWE agents, not evaluation of multi-turn belief drift / sycophancy / instability.
- Their “stability” is primarily optimization stability + task completion, rather than conversational robustness metrics (e.g., turn-of-failure, flip persistence).

## Where GALILEO is stronger / cleaner (if true)

- GALILEO-style protocols can provide **model-agnostic evaluation** of multi-turn instability without requiring RL training infrastructure.
- GALILEO can separate different failure modes (drift vs evidence updates vs social pressure), whereas G-RA is a training heuristic aimed at preventing collapse.

## Where GALILEO is weaker / needs to improve

- If GALILEO later uses RL-based tuning or auxiliary rewards (e.g., “stay consistent”/“don’t drift” stepwise checks), it risks the same **misalignment/collapse** issues without a principled accumulation strategy.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a short “reward misalignment” paragraph in related work: dense intermediate metrics can dominate sparse objectives; gating/masking is a practical mitigation.
- [ ] If we design any stepwise critics (format, calibration, tool-use correctness) during training, consider **gated accumulation** keyed on an outcome metric relevant to GALILEO.
- [ ] Consider borrowing their diagnostic: track outcome metric vs auxiliary metric trajectories to detect “metric hacking.”

## Quotes / details to potentially cite

- Abstract: “**Gated Reward Accumulation (G-RA)** … accumulates immediate rewards **only when high-level (long-term) rewards meet a predefined threshold**, ensuring stable RL optimization.”
- Reported gains (abstract): completion rates “**47.6% → 93.8%**” (SWE-bench Verified) and “**22.0% → 86.0%**” (kBench), with corresponding MR increases.
