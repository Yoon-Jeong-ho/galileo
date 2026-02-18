# Planning with Reasoning using Vision Language World Model

- Year: 2025
- Venue: arXiv
- Authors: Delong Chen, Pascale Fung, et al. (see arXiv for full list)
- URL: https://arxiv.org/abs/2509.02722
- BibTeX key (if we add it): Chen2025PlanningReasoningVLWM
- Tags: planning, reasoning, world-model, vision-language, video

## One-sentence takeaway

A vision-language world model predicts language-level action/state trajectories from video and supports both fast “system-1” rollout planning and a “system-2” cost-minimizing search using a learned critic.

## What problem does it solve?

- Pixel-/latent-video world models are expensive/ill-posed for long-horizon planning and often capture irrelevant details; meanwhile LLM/VLM prompting is not grounded and behavior cloning over web video can inherit suboptimal actions.
- Need a *high-level* (semantic + temporal abstraction) world model usable for planning and “reasoning” (search over imagined futures).

## What is the core method / protocol?

- **Vision Language World Model (VLWM):** a JEPA-style world model that predicts *structured textual abstractions* of the unobserved future rather than pixels.
- **Training target construction pipeline:**
  - Compress each video into a **Tree of Captions** (hierarchical captions over adaptively segmented temporal windows).
  - Use iterative **LLM Self-Refine** on the captions to produce structured trajectories containing:
    - goal description
    - goal interpretation (initial + desired final state)
    - interleaved **actions** and **world state changes** (ΔS)
- **Dual use at inference:**
  - **System-1 planning:** directly decode a plan/action sequence by autoregressive completion (policy-like).
  - **System-2 planning (“planning with reasoning”):** generate multiple candidate rollouts (candidate actions → predicted next states) and **search** for an action sequence minimizing a scalar **cost**.
    - Cost is computed by a **critic** LM trained self-supervised to score progress-toward-goal lower and irrelevant/counterfactual higher.

## What are the key metrics?

- Visual Planning for Assistance (VPA): success rate (SR), mean accuracy (mAcc), mean IoU (mIoU) (paper reports relative gains).
- PlannerArena: human preference / Elo score for plan quality.
- RoboVQA: BLEU-1.
- WorldPrediction procedural planning: accuracy.

## What are the main results?

- Claims SOTA on VPA benchmarks and their human eval:
  - VPA: reported relative gains about **+20% SR**, **+10% mAcc**, **+4% mIoU** (vs baselines, per intro).
  - PlannerArena: **system-2 improves Elo by +27%** over system-1.
- Also reports improvements on RoboVQA (e.g., **74.2 BLEU-1**) and WorldPrediction procedural planning (**45% accuracy**), plus strong critic performance incl. OOD.

## How is this similar to GALILEO?

- Uses an explicit **world-model + planner** separation: generate hypothetical futures and pick actions by evaluating them (model-based planning / cost minimization).
- Emphasizes **high-level, abstract state representations** rather than low-level pixels.
- System-2 framing aligns with “reasoning as search over imagined rollouts” (LeCun-style agent).

## How is this different from GALILEO?

- World state is **natural language** (interpretable but potentially brittle / underspecified) vs GALILEO’s likely more structured latent/state representation (depending on GALILEO design).
- Data/modeling focuses on **natural videos** and instruction/egocentric corpora, producing **procedural action + state-change text**, rather than environment-grounded simulators or robot-state trajectories.
- Planning is largely **LM decoding + text-critic scoring**, not necessarily grounded in executable constraints or verifiable dynamics.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO uses structured states / constraints / verification, it may avoid language underspecification and reduce hallucinated transitions.
- If GALILEO supports closed-loop execution, it may better handle distribution shift than offline web-video-derived trajectories.

## Where GALILEO is weaker / needs to improve

- If GALILEO lacks a strong “system-2” search layer (cost-minimizing over rollouts), VLWM-style **critic-guided search** could be a concrete upgrade path.
- If GALILEO’s abstraction is less interpretable, VLWM’s language-state could help debugging/analysis (even if only as an auxiliary representation).

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding a **system-2 mode**: sample multiple candidate action sequences, roll out with the world model, and select by a learned cost/critic.
- [ ] Explore **hybrid state representations**: structured + language “explanation” state for interpretability, trained as an auxiliary head.
- [ ] In related work, position GALILEO vs “language world models” that avoid pixel prediction by predicting **action/state-change text**.

## Quotes / details to potentially cite

- “VLWM … predicts a trajectory composed of interleaved actions and world state changes.”
- “System-2 planning via cost minimization … semantic distance between hypothetical future states … and the expected goal state.”
- Dataset scale (from intro): ~180k videos, 21M caption nodes, 1.2M extracted goal-plan trajectories (+ reformulated text-only reasoning trajectories).
