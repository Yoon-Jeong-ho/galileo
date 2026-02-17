# TSR: Trajectory-Search Rollouts for Multi-Turn RL of LLM Agents

- Year: 2026
- Venue: arXiv
- Authors: Aladin Djuhera (et al.; see arXiv)
- URL: https://arxiv.org/abs/2602.11767
- BibTeX key (if we add it): TSR2026TrajectorySearchRollouts
- Tags: agents, multi-turn, RL, rollouts, search, train-time-scaling

## One-sentence takeaway

TSR improves multi-turn RL for LLM agents by doing **lightweight per-turn search during rollout generation** (best-of-N / beam / shallow lookahead) to produce higher-quality trajectories, yielding steadier training and up to ~15% gains without changing the optimizer objective.

## What problem does it solve?

- Multi-turn RL for LLM agents is brittle because rewards are sparse/delayed and environments can be stochastic.
- Naively sampling full trajectories from the current policy yields many low-quality / unlucky rollouts, increasing variance and risking training instabilities / mode collapse (they reference “Echo Trap”).
- Question: if we spend a fixed extra compute budget at training-time rollout generation, can we materially improve stability + final performance?

## What is the core method / protocol?

- **TSR (Trajectory-Search Rollouts)**: replace “independent trajectory sampling” with a **tree-style search** over action sequences *during training rollouts*.
- At each turn, TSR considers multiple candidate actions (and optionally shallow futures), uses **task feedback** (e.g., per-turn reward / environment score signals) to pick higher-scoring actions, and constructs a higher-quality trajectory to feed into standard policy-gradient updates.
- Key point: TSR **does not change** the policy optimization objective; it only changes *how rollouts are generated*, so it is **optimizer-agnostic**.
- Instantiations mentioned:
  - **Best-of-N** per turn
  - **Beam search** across turns
  - **Shallow lookahead search** (limited depth)
- Combined with standard algorithms such as **PPO** and **GRPO**.

## What are the key metrics?

- Task success / performance in environments (reported as up to ~15% absolute gains vs baselines).
- Training stability / smoothness (qualitative claim: “more stable learning”, “smoother training dynamics”).

## What are the main results?

- On three environments spanning different regimes:
  - **Sokoban** (deterministic logic/puzzle)
  - **FrozenLake** (stochastic RL)
  - **WebShop** (long-horizon, agentic web navigation)
- TSR yields **up to ~15% performance gains** and **more stable learning** at a one-time increase in train-time compute.
- Claim: moving search from inference-time to *training rollout generation* is complementary to rejection-sampling-style selection methods.

## How is this similar to GALILEO?

- Shared focus on **multi-turn interaction trajectories** and how to evaluate/improve behavior over sequences rather than single-turn outputs.
- Emphasizes that **how you sample/construct trajectories** (not only the loss/optimizer) can strongly affect stability and outcomes—similar spirit to GALILEO’s interest in interaction dynamics.

## How is this different from GALILEO?

- TSR is about **training-time optimization for agents** (RL rollouts + policy updates), not primarily about **measurement protocols** for drift vs revision, social pressure, or belief stability in dialogue.
- Feedback used is **task-specific environment reward/score**, rather than social-pressure signals or “belief” correctness under persuasion.

## Where GALILEO is stronger / cleaner (if true)

- GALILEO can likely offer a cleaner story around **diagnosing multi-turn failures** (time-to-failure, recovery, drift-vs-revision controls) across dialogue-like pressures, independent of a particular RL environment reward.

## Where GALILEO is weaker / needs to improve

- If GALILEO proposes interventions/training, TSR suggests a concrete knob: **train-time rollout search/selection** as a stabilizer—GALILEO may need to address whether improved trajectory selection could confound comparisons (e.g., “stability” gains due to better rollouts rather than better policy).

## Action items for GALILEO (experiments / method / writing)

- [ ] Related-work paragraph: cite TSR as “**train-time test-time-scaling**” / search moved to the rollout stage for multi-turn RL of LLM agents.
- [ ] If we discuss training-time interventions: consider a baseline that does **best-of-N / beam** at rollout generation time to reduce instability in multi-turn training.
- [ ] If we report multi-turn robustness/stability: explicitly distinguish (i) stability from better **sampling/search** vs (ii) stability from better **policy/representation**.

## Quotes / details to potentially cite

- Abstract: “TSR performs lightweight tree-style search to construct high-quality trajectories by selecting high-scoring actions at each turn using task-specific feedback.”
- Abstract: “optimizer-agnostic” / “leaving the underlying optimization objective unchanged.”
- Abstract: “achieving up to 15% performance gains and more stable learning on Sokoban, FrozenLake, and WebShop tasks … by moving search from inference time to the rollout stage of training.”
