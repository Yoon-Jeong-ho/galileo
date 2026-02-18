# Offline RL by Reward-Weighted Fine-Tuning for Conversation Optimization

- Year: 2025
- Venue: NeurIPS 2025 (Neural Information Processing Systems 38)
- Authors: Subhojyoti Mukherjee; Viet Dac Lai; Raghavendra Addanki; Ryan Rossi; Seunghyun Yoon; Trung Bui; Anup Rao; Jayakumar Subramanian; Branislav Kveton
- URL: https://arxiv.org/abs/2506.06964
- BibTeX key (if we add it): Mukherjee2025OfflineRLRewardWeightedFT
- Tags: multi-turn; offline-RL; reward-weighted fine-tuning; conversation optimization; clarifying questions; RLHF-adjacent

## One-sentence takeaway

Offline RL for LLM conversation policies can be implemented as *trajectory-level reward-weighted fine-tuning* (no PPO-style ratios), yielding more stable optimization and better reward/language quality than SFT/DPO baselines on multi-turn QA tasks.

## What problem does it solve?

- How to do *practical offline RL* for LLM-based multi-turn conversation policies using only logged trajectories + scalar rewards, without fragile RL machinery (e.g., token-level propensity ratios) and with a training loop close to standard SFT.
- Target setting: short-horizon, fixed-length (or variable-stop) multi-turn QA where the agent may reason, answer, or ask clarifying questions before finalizing.

## What is the core method / protocol?

- Recast offline RL as maximizing a *lower bound* on the online RL objective.
- The resulting objective becomes **reward-weighted fine-tuning**:
  - Weight the log-probability of the *logged action sequence* (trajectory of tokens/actions) by the trajectory reward.
  - Importantly, weights are over **sequences**, not per-token weights.
- Algorithms:
  - **Refit**: optimize the reward-weighted objective directly (designed to avoid propensity-score ratios typical in PPO-like methods).
  - **Swift**: a Refit variant that **standardizes rewards** across multiple logged trajectories (variance reduction; inspired by GRPO-style normalization).

## What are the key metrics?

- Primary: the optimized scalar **reward** used for training.
- Reported additional quality axes (as described in the paper): language quality and task-relevant metrics such as reasoning ability / pedagogical value / confidence (depending on benchmark and evaluator setup).

## What are the main results?

- Refit/Swift outperform SFT and DPO-style baselines in both:
  - optimized rewards, and
  - language quality,
  on a suite of multi-turn QA tasks.
- Claimed benefit: direct reward optimization avoids “information loss” incurred by reducing reward learning to preference-style or filtered-SFT objectives.

## How is this similar to GALILEO?

- If GALILEO involves multi-turn agent behavior optimization with reward signals, this is a close methodological neighbor: it proposes a *stable, SFT-like* way to incorporate trajectory-level rewards.
- Emphasizes evaluation beyond a single metric: they optimize one reward but observe improvements across other quality metrics.

## How is this different from GALILEO?

- Focus is offline RL for *conversation/QA policies* with trajectory rewards; not primarily about robustness-to-misleading prompts / drift / recovery unless GALILEO’s reward/eval explicitly targets those.
- Uses a specific theoretical lower-bound justification and produces Refit/Swift objectives tailored to logged trajectories.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO’s contribution is in *evaluation protocols for multi-turn robustness (time-to-failure, recovery)*, that is largely orthogonal to this paper’s main novelty (optimization objective/training recipe).

## Where GALILEO is weaker / needs to improve

- If GALILEO currently relies on SFT/DPO-like objectives for trajectory optimization, this paper suggests a more “RL-faithful but SFT-like” alternative (reward-weighted trajectory objective + reward standardization) that may be simpler than PPO variants.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding a baseline/objective: **trajectory-level reward-weighted fine-tuning** (Refit-style) as a simpler offline-RL training variant.
- [ ] If using reward normalization, test a **Swift-style reward standardization** (variance reduction) and report stability.
- [ ] In related work, contrast with PPO/GRPO by highlighting **no propensity-score ratios** + **sequence-level** weighting.

## Quotes / details to potentially cite

- “We recast the problem as reward-weighted fine-tuning, which can be solved using similar techniques to supervised fine-tuning (SFT).”
- Refit motivation: avoids “ratios of token-level propensity scores, unlike in PPO and GRPO,” aiming for a stable, practical algorithm.
- Claimed key distinction vs prior offline RL for LLM post-training: weights are over **sequences of tokens**, not individual tokens.
