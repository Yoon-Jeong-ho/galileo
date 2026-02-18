# Embodied Navigation Foundation Model

- Year: 2025
- Venue: arXiv
- Authors: Jiazhao Zhang, Anqi Li, Yunpeng Qi, Minghan Li, Jiahang Liu, Shaoan Wang, Haoran Liu, Gengze Zhou, Yuze Wu, Xingxing Li, Yuxin Fan, Wenjun Li, Zhibo Chen, Fei Gao, Qi Wu, Zhizheng Zhang, He Wang
- URL: https://arxiv.org/abs/2509.12129
- BibTeX key (if we add it): navfom2025
- Tags: navigation, embodied-ai, foundation-model, cross-embodiment, cross-task, vlm

## One-sentence takeaway

NavFoM trains a single vision-language navigation model across embodiments (quadruped/drone/wheeled/car) and tasks (VLN, object search, tracking, driving) using viewpoint/time “identifier” tokens plus budget-aware temporal token sampling to fit a fixed context window.

## What problem does it solve?

- Existing VLM/LLM-based navigation systems tend to be (a) task-specific, (b) embodiment/camera-setup-specific, and/or (c) expensive at inference due to long histories.
- Goal: a unified navigation “foundation model” that generalizes across tasks and embodiments without per-task fine-tuning, while respecting token budget constraints in deployment.

## What is the core method / protocol?

- Formulation: predict future trajectories given egocentric visual history (one or more cameras) + language instruction.
- **TVI (temporal-viewpoint indicator) tokens**: identifier tokens encode camera viewpoint/configuration + temporal context/horizon so the model can disentangle tokens from different cameras/timesteps.
- **BATS (Budget-Aware Temporal Sampling)**: dynamically sample observation/history tokens under a fixed token budget (described as based on a forgetting-curve idea) to trade off performance vs latency/memory.
- Training data: ~8M navigation samples across multiple embodiments/tasks, plus additional image/video QA samples for co-tuning (intended to improve open-world knowledge and generalization).

## What are the key metrics?

- Success rate (SR) and benchmark-specific navigation metrics (paper reports SR prominently; other tasks likely use task-standard metrics).
- Emphasis on *zero-shot / no task-specific fine-tuning* comparisons.

## What are the main results?

- Reported SOTA or competitive results across **seven** public benchmarks spanning multiple navigation task types.
- Example numbers highlighted in the intro:
  - VLN-CE RxR: SR improves from ~56.3% to ~64.4% (multi-camera) and ~51.8% to ~57.4% (single-camera) versus prior baselines.
  - HM3D-OVON: zero-shot SR ~45.2%, exceeding a prior fine-tuned SOTA (~43.6%).
- Includes real-world robot experiments across multiple platforms (humanoid, quadruped, drone, wheeled) to support practical applicability.

## How is this similar to GALILEO?

- Shared motivation: **generalization** across settings rather than narrow, benchmark-tuned policies.
- Uses **tokenization/conditioning tricks** to make a single model handle heterogeneous inputs (camera setups, temporal horizons) within a fixed-context budget.
- Frames navigation as sequence modeling from multimodal context toward action/trajectory prediction.

## How is this different from GALILEO?

- Primary contribution is a **cross-embodiment, cross-task navigation foundation model** + dataset scale (millions of navigation samples) and deployment token budgeting.
- Focus is explicitly on navigation trajectories across robot types; GALILEO may be centered on different capabilities (depending on our paper’s core claim) or broader agentic/world-model objectives.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO emphasizes principled world modeling / planning / uncertainty / interpretability, it may provide a clearer conceptual story than “token tricks + scale” for unifying tasks.
- If GALILEO uses a cleaner modularization between perception, memory, and control, it may be easier to analyze than a monolithic foundation model.

## Where GALILEO is weaker / needs to improve

- If GALILEO targets navigation generalization, NavFoM is a strong comparison point on **cross-embodiment + multi-camera + token-budgeted history**.
- Consider whether GALILEO needs an explicit strategy analogous to BATS for long-horizon history under tight context limits.

## Action items for GALILEO (experiments / method / writing)

- [ ] Related work: cite NavFoM as a representative *cross-embodiment, cross-task navigation foundation model* trained at multi-million sample scale.
- [ ] Writing: explicitly discuss the “identifier token” approach for multi-view + temporal disambiguation; contrast with GALILEO’s representation (or explain why GALILEO avoids this).
- [ ] Method idea: evaluate a simple **budget-aware temporal sampling** baseline for GALILEO’s visual/history tokens (fixed token budget) to see if similar gains appear.
- [ ] If we claim generalist behavior: clarify whether we support cross-embodiment transfer and multi-camera setups; if not, scope the claim.

## Quotes / details to potentially cite

- “We introduce a cross-embodiment and cross-task Navigation Foundation Model (NavFoM), trained on eight million navigation samples … spanning diverse tasks such as vision-and-language navigation, object searching, target tracking, and autonomous driving.”
- “NavFoM incorporates identifier tokens that embed camera view information of embodiments and the temporal context of tasks.”
- “NavFoM controls all observation tokens using a dynamically adjusted sampling strategy under a limited token length budget.”
