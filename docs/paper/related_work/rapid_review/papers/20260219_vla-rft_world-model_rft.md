# VLA-RFT: Vision-language-action Reinforcement fine-tuning with verified rewards in world simulators

- Year: 2025
- Venue: arXiv
- Authors: Hengtao Li, Pengxiang Ding, Runze Suo, Yihao Wang, Zirui Ge, Dongyuan Zang, Kexian Yu, Mingyang Sun, Hongyin Zhang, Donglin Wang, Weihua Su
- URL: https://arxiv.org/html/2510.00406v1
- BibTeX key (if we add it): vla_rft_2025
- Tags: vla, reinforcement-finetuning, world-model, verified-reward, grpo

## One-sentence takeaway

Uses a learned action-conditioned video world model as a controllable simulator to do GRPO-style reinforcement fine-tuning of a VLA policy, with “verified” dense rewards derived from trajectory similarity, yielding gains with ~400 RFT steps and improved robustness to perturbations.

## What problem does it solve?

- Pure imitation-learned VLA policies suffer from compounding errors and brittleness under distribution shift.
- RL could improve robustness but is expensive (real robots) or suffers sim-to-real + high interaction counts (classical simulators).
- Need a practical post-training recipe to improve VLA robustness using mainly offline data.

## What is the core method / protocol?

- Train a **data-driven world model** on offline robot interaction trajectories (initial frame + action sequence → predicted future frames), positioned as an “interactive simulator”.
- Pretrain a VLA policy (vision-language encoder + flow-matching action head) via supervised learning for stable action chunks.
- Reinforcement fine-tune the VLA by:
  - Sampling multiple rollouts of action chunks from a stochasticized flow-matching policy (adds a “Sigma Net” to define per-step action variance).
  - Rolling out predicted visual trajectories in the world model.
  - Computing a dense, trajectory-level **verified reward** based on similarity between world-model-generated trajectories and goal-achieving reference trajectories (the paper emphasizes designs that compare within the world-model’s generative space to reduce bias from imperfect generation).
  - Optimizing with **GRPO** (group-relative policy optimization), using group-average reward as a baseline, plus an auxiliary flow-matching MSE term.

## What are the key metrics?

- Primary: task **success rate (SR)** on LIBERO suites.
- World-model fidelity: MSE / PSNR / SSIM / LPIPS on predicted frames.
- Robustness: SR under systematic perturbations (object position, goal position, robot state, combined).
- Efficiency headline: number of RFT iterations/steps (they emphasize ~400 vs very large supervised iteration counts).

## What are the main results?

- On LIBERO standard suites, ~400 RFT iterations improves average SR vs a strong SFT baseline (reported ~+4.5 SR points on their chosen baseline), with gains across suites.
- Under perturbations, RFT improves SR relative to SFT, especially for goal-position and combined perturbations (several-point gains).
- Reward-design ablations suggest trajectory-comparison-in-WM-space is substantially stronger than action-distance-only or direct WM-vs-real image comparison.

## How is this similar to GALILEO?

- Shares a core post-training theme: **optimize a learned policy using a reward signal derived from comparing model behavior to reference behavior**, aiming to improve robustness/generalization with limited additional training.
- Uses group-relative baselines (GRPO) and emphasizes robustness under perturbation/out-of-distribution settings.

## How is this different from GALILEO?

- Domain: embodied robotics VLA fine-tuning; GALILEO is not focused on action-conditioned video world models.
- Reward construction is heavily tied to **trajectory similarity to demonstrations** (still largely “demonstration-anchored”), rather than truly open-ended task reward.
- Relies on training and rolling out a **world model** to generate synthetic observation trajectories; GALILEO’s setting/methodology likely doesn’t assume a learned environment model.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO’s reward/target is not anchored to demonstration similarity, GALILEO may make a cleaner claim about improving behavior beyond imitation rather than “imitation in WM space”.
- If GALILEO does not require training a high-dimensional generative world model, it may be simpler/cheaper to reproduce.

## Where GALILEO is weaker / needs to improve

- If GALILEO lacks a simulator/world-model component, it may have fewer knobs to generate dense trajectory-level feedback signals.
- If GALILEO uses only sparse/terminal signals, this paper is a reminder that **dense trajectory-level rewards** (even if proxy) can dramatically improve sample efficiency.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider whether any GALILEO evaluation could benefit from a **trajectory-level reward** rather than turn/step-local rewards (e.g., compare full rollouts against a “goal-achieving reference” sequence).
- [ ] If GALILEO uses RL-style objectives, consider citing GRPO-style group baselines as a stability/variance-reduction analogy.
- [ ] In writing, explicitly discuss the tradeoff between (i) demonstration-anchored “verified” rewards and (ii) true task rewards; this paper is a concrete example of the former.

## Quotes / details to potentially cite

- Motivation framing: imitation learning in VLAs → compounding errors under distribution shift; RL helps but real-world cost + sim-to-real gap.
- Headline claim: “With fewer than 400 fine-tuning steps, VLA-RFT surpasses strong supervised baselines” and improves robustness under perturbations.
- Verified reward design: strongest variant compares both policy and reference trajectories within the world model’s generative space and uses MAE + LPIPS over time.
