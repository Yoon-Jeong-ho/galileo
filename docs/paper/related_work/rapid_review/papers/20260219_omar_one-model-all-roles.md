# One Model, All Roles: Multi-Turn, Multi-Agent Self-Play Reinforcement Learning for Conversational Social Intelligence

- Year: 2026
- Venue: arXiv (cs.CL)
- Authors: Bowen Jiang; Taiwei Shi; Ryo Kamoi; Yuan Yuan; Camillo J. Taylor; Longqi Yang; Pei Zhou; Sihao Chen
- URL: https://arxiv.org/abs/2602.03109
- BibTeX key (if we add it): jiang2026omar
- Tags: multi-agent; self-play; conversational-RL; multi-turn; PPO; social-intelligence; Sotopia; werewolf

## One-sentence takeaway

OMAR trains *one* policy model to role-play all participants in a multi-turn conversation (self-play) and uses hierarchical (turn+token) advantage estimation to stabilize PPO on long-horizon dialogue, yielding improved “social intelligence” behaviors in Sotopia and Werewolf.

## What problem does it solve?

- Standard LLM post-training RL (e.g., RLVR/GRPO-style) is typically **single-turn** and/or assumes independent rollouts from the same prompt, which does not match **multi-turn, multi-agent** social interaction.
- Directly applying PPO to long multi-turn dialogues yields **high-variance** credit assignment when only a terminal reward is available.

## What is the core method / protocol?

- **OMAR (One Model, All Roles)**: reinterpret the “n rollouts per prompt” idea as “n participants in a group conversation.”
  - At each turn *t*, the same actor model produces utterances for each active role/persona in parallel.
  - The environment aggregates all utterances from turn *t* into the shared context for turn *t+1*.
  - Rewards are **end-of-episode** and environment-dependent (e.g., satisfaction/goal metrics in Sotopia; win/loss in Werewolf).
- **Optimization**: use **PPO** (not GRPO normalization), because the participants’ trajectories are coupled and not independent.
- **Hierarchical advantage estimation (key stabilization idea)**:
  - Turn-level: treat each turn as one step; approximate turn value by the value at the last token of the turn; compute turn advantages via GAE from the terminal reward.
  - Token-level: treat the turn advantage as a pseudo-reward at end-of-turn; run within-turn token-level GAE to get token advantages.
- **Reward hacking mitigation** (practical): cold-start SFT + “turn-level quality filtering” (LLM-as-judge filters degenerate turns; filtered turns get zero pseudo-reward) + early stopping.

## What are the key metrics?

- **Sotopia**: 7 criteria (goal completion, believability, knowledge, secret compliance, social rule compliance, relationship, financial benefit), aggregated as terminal reward per participant.
- Additional **fine-grained social intelligence metrics** (0–10) used for zero-shot evaluation: compromise seeking, persuasion skill, strategic commitment, empathy expression, mutual benefit seeking.
- **Werewolf**: terminal win/loss reward (with discount for early-eliminated players); behavior probes like identity concealment / voting manipulation / intra-team collaboration (and villager-side collaboration behaviors).

## What are the main results?

- In Sotopia arena-style evaluation (trained model vs base model assigned to roles at inference), OMAR-trained Qwen-2.5-7B shows **consistent improvements** on reward-aligned Sotopia metrics and **larger gaps** on the proposed fine-grained social-skill metrics.
- In Werewolf with Qwen-3-4B, training on only win/loss outcomes yields emergence of socially strategic behaviors (concealment, manipulation, collaboration) and increases werewolf-team win rate (paper reports ~55% → ~72%).
- Authors emphasize that meaningful social behaviors can emerge **without direct human supervision**, but training is fragile due to reward hacking.

## How is this similar to GALILEO?

- Shared theme: **multi-agent interaction** and learning behaviors that look like “social” competence (coordination, persuasion, compromise) from interaction signals.
- Terminal / outcome-driven rewards + long-horizon credit assignment issues are likely analogous if GALILEO uses episode-level objectives.

## How is this different from GALILEO?

- OMAR focuses on **self-play with one model controlling all roles** during training (batch = roles), which is a specific training construction; GALILEO may instead focus on multi-agent systems with distinct policies / external agents / different supervision.
- Their stabilization contribution is primarily **advantage estimation structure** for PPO on long dialogues, plus quality filtering to suppress degenerate turns.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO uses explicit environment state, better-defined credit assignment, or richer intermediate signals, it may avoid some of OMAR’s reliance on an LLM judge and ad-hoc filtering.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently applies single-turn RLHF/RLVR-style updates for social tasks, OMAR is a reminder that **multi-turn coupling** breaks “independent rollout” assumptions and may require different advantage / normalization choices.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider a “one-model-multi-role” ablation: same base policy, but training batches structured as **roles within the same conversation**, not independent rollouts.
- [ ] If training is unstable with terminal rewards, prototype **hierarchical (turn/token) advantage estimation** to reduce variance.
- [ ] Add an explicit “reward hacking & degeneration” section: quality filters, early stopping signals (entropy growth), and examples of failure.
- [ ] For evaluation, add **fine-grained social behavior metrics** (compromise/persuasion/empathy/commitment) beyond high-level aggregate scores.

## Quotes / details to potentially cite

- “OMAR allows a single model to role-play all participants in a conversation simultaneously, learning to achieve long-term goals and complex social norms directly from dynamic social interaction.”
- “To ensure training stability across long dialogues, we implement a hierarchical advantage estimation that calculates turn-level and token-level advantages.”
- Reported Werewolf result: werewolf win rate “~55% to ~72%” after training (check final camera-ready numbers if citing).
