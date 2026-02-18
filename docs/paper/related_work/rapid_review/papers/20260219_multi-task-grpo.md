# Multi-Task GRPO: Reliable LLM Reasoning Across Tasks

- Year: 2026
- Venue: arXiv (preprint)
- Authors: Shyam Sundhar Ramesh; Xiaotong Ji; Matthieu Zimmer; Sangwoong Yoon; Zhiyong Wang; Haitham Bou Ammar; Aurelien Lucchi; Ilija Bogunovic
- URL: https://arxiv.org/abs/2602.05547
- BibTeX key (if we add it): ramesh2026mtgrpo
- Tags: robustness, post-training, RL, worst-case, multitask, GRPO

## One-sentence takeaway

MT-GRPO modifies multi-task GRPO with (1) improvement-aware task reweighting to optimize worst-task performance and (2) a ratio-preserving sampler to keep realized gradients aligned with desired task weights despite task-dependent zero-gradient rates.

## What problem does it solve?

- Naive multi-task GRPO tends to be *imbalanced*: “easy” tasks dominate optimization, while harder tasks stagnate, so worst-task accuracy stays low even if average accuracy rises.
- A GRPO-specific issue: tasks differ in how often prompts yield **zero advantages / zero gradients** (e.g., all sampled rollouts get identical rewards), which implicitly reweights tasks away from the intended mixture.
- Strict worst-task (minimax) reweighting can collapse weights onto a single worst task and oscillate, under-optimizing others.

## What is the core method / protocol?

- **Objective framing:** robustness-aware multi-task RL post-training, emphasizing *worst-task* performance (a DRO / max–min flavored formulation with a tunable trade-off vs average performance).
- **IWU (Improvement-aware Weight Update):** update task weights using a combined signal of:
  - task-level rewards (true per-task performance, not the ambiguous GRPO loss), and
  - *task-level improvement* based on change in the task-wise GRPO objective across steps.
  This aims to prioritize tasks that are both underperforming and under-improving.
- **RP sampler (Ratio-Preserving sampler):** acceptance-/filtering-aware batch construction to enforce target task proportions *after* filtering zero-gradient prompts, via oversampling and resampling guided by estimated per-task filtering rates.
- Evaluated by post-training (example setup) **Qwen-2.5-3B** on multi-task reasoning benchmarks spanning planning + inductive reasoning (Countdown, Zebra, ARC), in 3-task and 9-task variants.

## What are the key metrics?

- **Worst-task accuracy** (minimum accuracy across tasks) as the primary robustness metric.
- Average accuracy across tasks.
- “Average per-task relative change” (a normalized improvement metric that emphasizes gains on weaker tasks).
- Training efficiency: steps to reach target worst-task accuracy thresholds.

## What are the main results?

- Across 3-task and 9-task experiments, MT-GRPO improves **worst-task accuracy** versus strong baselines (reported: +16–28% absolute over standard GRPO; +6% absolute over DAPO, in some settings).
- **Efficiency:** reported ~50% fewer training steps to reach 50% worst-task accuracy in the 3-task setup.
- Empirical analyses attribute gains to:
  - adaptive weight reallocation toward weaker/stagnating tasks (vs baselines that keep oversampling already-strong tasks), and
  - ratio-preservation preventing high-filtering tasks (e.g., ARC) from being underrepresented in realized gradient updates.

## How is this similar to GALILEO?

- Shares the motivation of **robustness / reliability** rather than just average performance: explicitly optimizing for the worst-performing slice/task.
- Uses dynamic reweighting to allocate optimization effort where it improves reliability.

## How is this different from GALILEO?

- This paper is specifically about **multi-task RL post-training with GRPO-style objectives**, focusing on:
  - GRPO loss ambiguity for cross-task comparisons, and
  - task-dependent **zero-gradient** rates that distort effective optimization.
- The key technical contribution is an *algorithmic training pipeline* (IWU + RP sampler), not (as far as this review can tell from skim) a new inference-time reasoning protocol.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO targets broad reliability beyond verifiable-task RL settings, it may generalize more naturally to tasks without clean reward functions / filtering heuristics.
- If GALILEO’s design avoids per-task sampling engineering, it may be simpler to implement or reason about.

## Where GALILEO is weaker / needs to improve

- If GALILEO also uses multi-task optimization, it may face similar “hidden reweighting” issues when some tasks yield weaker/rarer learning signals; this work suggests explicitly measuring and correcting those distortions.
- If GALILEO’s evaluation emphasizes mean performance, consider adding worst-case / tail metrics as first-class objectives.

## Action items for GALILEO (experiments / method / writing)

- [ ] In multi-task settings, report **worst-task** (or worst-slice) metrics alongside average; consider a tunable trade-off parameter similar in spirit to ε/λ.
- [ ] Check for **task-dependent zero-signal / zero-gradient** phenomena (or analogues) that can silently skew optimization; add diagnostics (e.g., per-task “useful update rate”).
- [ ] If GALILEO uses dynamic mixture weights, ensure realized update contributions match intended weights (e.g., post-filter ratio preservation / acceptance-aware sampling).

## Quotes / details to potentially cite

- Abstract (problem): “A straightforward multi-task adaptation of GRPO often leads to imbalanced outcomes, with some tasks dominating optimization while others stagnate.”
- Abstract (zero-gradient issue): “tasks can vary widely in how frequently prompts yield zero advantages (and thus zero gradients), which further distorts their effective contribution to the optimization signal.”
- Abstract (method): “(i) dynamically adapts task weights to explicitly optimize worst-task performance … and (ii) introduces a ratio-preserving sampler to ensure task-wise policy gradients reflect the adapted weights.”
- Intro framing: optimizing average reward “permits solutions in which strong gains on a subset of tasks compensate for substantial underperformance on others…”.
