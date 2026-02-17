# Reinforcing Multi-Turn Reasoning in LLM Agents via Turn-Level Reward Design

- Year: 2025
- Venue: arXiv
- Authors: Quan Wei; Siliang Zeng; Chenliang Li; William Brown; Oana Frunza; Wei Deng; Anderson Schneider; Yuriy Nevmyvaka; Yang Katie Zhao; Alfredo Garcia; Mingyi Hong
- URL: https://arxiv.org/abs/2505.11821
- BibTeX key (if we add it): wei2025turnlevelreward (suggested)
- Tags: multi-turn, agents, RL, GRPO, PPO, reward-design, credit-assignment, search-agent

## One-sentence takeaway

Designing *turn-level* intermediate rewards (verifiable or LLM-judge) for multi-turn GRPO/PPO yields more stable training and higher accuracy than trajectory-only outcome rewards for a reasoning+search agent.

## What problem does it solve?

- Multi-turn LLM agents (tool-using / interactive) are typically trained with RL using sparse final outcome rewards (e.g., final answer correctness).
- Sparse, end-only rewards make credit assignment across many turns difficult, leading to unstable training and weaker multi-step behavior (e.g., poor early search query choices).

## What is the core method / protocol?

- Formulate agent interaction as a *turn-level MDP* where each turn corresponds to an agent action and (optional) environment feedback.
- Extend common RL algorithms to multi-turn variants by incorporating intermediate per-turn rewards:
  - Multi-turn GRPO: uses group-relative optimization but (as noted) can require many rollouts to compute intermediate advantages.
  - Multi-turn PPO: uses a critic to estimate values/advantages turn-by-turn (more scalable).
- Case study: a “reasoning-augmented search agent” that alternates reasoning and search over multiple turns before producing a final answer.
- Turn-level reward designs:
  - **Verifiable** rewards: rule-based/automatic signals tied to intermediate steps.
  - **LLM-as-judge** rewards: an LLM evaluates intermediate steps more flexibly.

## What are the key metrics?

- Final answer correctness (accuracy) on multi-turn QA/search tasks.
- Training/validation reward curves (stability and convergence speed).
- Output format correctness (reported as 100% with their approach).

## What are the main results?

- Adding well-designed turn-level rewards substantially outperforms trajectory-level (final-only) reward baselines on multi-turn search tasks.
- Reported benefits include:
  - greater training stability,
  - faster convergence,
  - higher accuracy,
  - consistent improvements across multiple QA datasets,
  - 100% format correctness.
- Experiments mention Qwen2.5-7B as the base model for reward curves/benchmarks.

## How is this similar to GALILEO?

- Shares the framing that **multi-turn interaction should be modeled/assessed at the turn level**, not collapsed into a single end-of-trajectory scalar.
- Emphasizes that intermediate-turn signals (rewards/feedback) matter for understanding and improving multi-step agent behavior.

## How is this different from GALILEO?

- This work is primarily about **training** agents with RL (GRPO/PPO) using turn-level rewards.
- GALILEO (as a paper direction) is primarily about **evaluating robustness dynamics** over multi-turn interactions (e.g., failures over rounds, drift baselines), rather than proposing RL training objectives.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO’s protocol cleanly separates *generic multi-turn drift* from *condition-induced effects* (e.g., via an explicit control condition), that’s a complementary strength: it can diagnose failure dynamics even without changing training.

## Where GALILEO is weaker / needs to improve

- If GALILEO aims to provide actionable fixes (not just diagnosis), this paper suggests a concrete lever: **designing turn-level feedback/rewards** aligned to intermediate steps.

## Action items for GALILEO (experiments / method / writing)

- [ ] Related-work positioning: cite as evidence that the community is moving from “trajectory-only” to **turn-level** formulations for multi-turn agents; contrast *training with turn rewards* vs *evaluating robustness dynamics*.
- [ ] Consider adding a short paragraph in related work on **turn-level supervision** (verifiable vs LLM-judge) as a general mechanism to shape multi-turn behavior.

## Quotes / details to potentially cite

- “Although RL algorithms such as GRPO and PPO … typically rely only on sparse outcome rewards and lack dense intermediate signals across multiple decision steps…”
- “By integrating turn-level rewards, we extend GRPO and PPO to their respective multi-turn variants, enabling fine-grained credit assignment.”
- “Two types of turn-level rewards: verifiable and LLM-as-judge.”
