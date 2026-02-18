# From Seeing to Experiencing: Scaling Navigation Foundation Models with Reinforcement Learning

- Year: 2025
- Venue: arXiv
- Authors: Honglin He; Yukai Ma; Wayne Wu; Bolei Zhou (equal contribution noted)
- URL: https://arxiv.org/abs/2507.22028
- BibTeX key (if we add it): He2025SeeingToExperiencing
- Tags: navigation, foundation-model, reinforcement-learning, sim2real, safety, benchmark

## One-sentence takeaway

Pretrain a navigation foundation model on large-scale real-world video, then *post-train with RL in simulation* using a stabilization/distillation-style scheme (AGDM + residual attention) to improve interactivity/safety without losing generalization, and evaluate on a new photorealistic 3DGS-based benchmark (NavBench-GS).

## What problem does it solve?

- Purely offline / video-pretrained navigation policies generalize visually but often fail to be *reactive* and *safe* in interactive urban settings (dynamic pedestrians/obstacles, counterfactuals, cause-effect).
- RL in sim can add interactivity, but naive RL tends to be sample-inefficient and can overfit / forget pretrained priors.
- Lack of evaluation that stresses both *generalization* and *physical interaction safety* in realistic reconstructions.

## What is the core method / protocol?

- Seeing-to-Experiencing (S2E) framework:
  - Offline pretraining on real-world videos.
  - Post-training with reinforcement learning in simulation to add reactive behaviors.
- Two technical components called out in the paper:
  - **Anchor-Guided Distribution Matching (AGDM)** for pretraining: uses anchor-based supervision to stabilize learning and model multimodal motion/trajectory distributions; also intended to support cross-embodiment deployment.
  - **Residual-Attention Module (RAM)** for RL post-training: a residual/adapter-like attention module intended to inject reactive behaviors from sim *without erasing* the pretrained knowledge (mitigate catastrophic forgetting during RL).
- Benchmark:
  - **NavBench-GS**: end-to-end evaluation on photorealistic **3D Gaussian Splatting** reconstructions of real-world scenes with physical interactions, to test generalization + safety.

## What are the key metrics?

(From positioning; exact numbers not captured in a rapid skim)

- Navigation task success / goal-reaching rate.
- Safety: collision rate / near-collision / interaction violations with dynamic obstacles/pedestrians.
- Generalization: zero-shot transfer across environments and embodiments (they mention wheeled + quadruped deployments).
- Potentially efficiency metrics (trajectory length, time-to-goal) and intervention rate.

## What are the main results?

- S2E reportedly mitigates the **diminishing returns** from scaling offline data alone by adding online interactive learning.
- RL post-training with the proposed RAM is claimed to improve reactivity/safety while preserving pretrained generalization.
- Paper emphasizes analysis comparing **RL vs supervised fine-tuning (SFT)** for post-training in robot learning, arguing RL brings benefits for interactive behaviors.

## How is this similar to GALILEO?

- Same high-level motivation: build navigation/robot policies that generalize beyond a single environment by leveraging large-scale pretraining and stronger post-training.
- Focus on safety/robustness in real-world-like urban navigation settings.

## How is this different from GALILEO?

- Explicit **RL post-training** recipe (and architecture modifications) is central; if GALILEO is primarily offline/self-supervised (or uses different alignment), this is a distinct route.
- Introduces an evaluation benchmark based on **3DGS reconstructions with physical interactions** (NavBench-GS), which may differ from GALILEO's evaluation stack.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO avoids heavy simulator-specific RL and/or uses a simpler adaptation mechanism, it may have a cleaner training pipeline and fewer sim-to-real assumptions.

## Where GALILEO is weaker / needs to improve

- If GALILEO lacks interactive post-training, this paper is a concrete reference for *how to add interactivity* while attempting to prevent forgetting.
- Benchmarking: NavBench-GS suggests a direction for more systematic safety + interaction evaluation.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add related-work paragraph: “offline video pretraining is not enough; RL post-training can add interactivity” with S2E as a representative approach.
- [ ] Consider an ablation or discussion contrasting **RL post-training vs SFT** for any GALILEO post-training stage.
- [ ] Consider whether a **residual adapter** (residual attention / LoRA-like) during interactive fine-tuning can reduce forgetting.
- [ ] Track evaluation desiderata: include an interaction-heavy benchmark (dynamic obstacles) and report safety metrics explicitly.

## Quotes / details to potentially cite

- “From seeing to experiencing”: offline video captures correlations but not grounded causality/counterfactuals; interactive experience is needed for adaptability.
- Contributions list (as stated on the arXiv HTML abstract):
  - Anchor-Guided Distribution Matching (AGDM) for stabilizing offline pretraining and modeling diverse motion patterns.
  - Residual-Attention Module (RAM) to obtain reactive behaviors via RL without erasing pretrained knowledge.
  - NavBench-GS benchmark using photorealistic 3D Gaussian Splatting reconstructions with physical interactions.
