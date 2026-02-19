# A Survey of Robotic Navigation and Manipulation with Physics Simulators in the Era of Embodied AI

- Year: 2025
- Venue: arXiv (survey; notes mention CSUR)
- Authors: Lik Hang Kenny Wong; Xueyang Kang; Kaixin Bai; Jianwei Zhang
- URL: https://arxiv.org/abs/2505.01458
- BibTeX key (if we add it): wong2025survey_robotic_simulators
- Tags: survey, embodied-ai, simulators, navigation, manipulation, sim2real

## One-sentence takeaway

A broad 2025 survey that frames navigation+manipulation progress through the lens of physics simulators and sim-to-real gaps, with a catalog of simulators/datasets/metrics and emerging directions (world models, geometric equivariance).

## What problem does it solve?

- Consolidates fast-moving embodied navigation/manipulation literature and (unlike earlier surveys) emphasizes simulator properties that matter for sim-to-real (visual realism, physics realism, differentiability, hardware cost).
- Provides a “buyer’s guide” style resource: which simulators/benchmarks/methods map to which task requirements.

## What is the core method / protocol?

- Not a new algorithm; survey taxonomy + structured comparison.
- Organizes embodied navigation into a 4-step loop (perception → memory building → decision making → action execution) and ties simulator requirements to each step and to two gaps:
  - visual sim-to-real gap (rendering realism, sensor noise models)
  - physics sim-to-real gap (dynamics fidelity: collision, friction, uneven terrain, etc.)
- Categorizes navigation simulators by environment scale/type (indoor vs outdoor vs general-purpose) and discusses common stacks (Unity/Unreal, Bullet/PhysX, Habitat/AI2-THOR/CARLA/AirSim/TDW/Isaac Sim, etc.).

## What are the key metrics?

- Survey-level: success rate (SR) and task-specific metrics across standard navigation/manipulation benchmarks.
- Emphasis on benchmark properties (scene scale, sensor modalities like RGB/Depth/Segmentation, availability of expert trajectories) more than proposing new metrics.

## What are the main results?

- No experimental SOTA claims; main “results” are the taxonomy + comparative tables and the argument that simulator choice (and its realism vs throughput vs hardware trade-offs) is a first-class design decision for embodied learning.

## How is this similar to GALILEO?

- High-level connection: both care about *evaluation conditions* and *environmental/interaction fidelity* (their domain: embodied physical interaction; ours: multi-turn interaction robustness/consistency).
- Useful as a citation if we want to position “interactive evaluation” as spanning text-only and embodied settings, with simulators as the analogue of “test harnesses”.

## How is this different from GALILEO?

- Different problem domain (robotics/embodiment) and different technical focus (physics/rendering engines, sim-to-real transfer, RL/IL/VLA policies).
- Does not address LLM multi-turn failure dynamics / turn-of-failure style metrics.

## Where GALILEO is stronger / cleaner (if true)

- If we frame GALILEO as an *auditable evaluation protocol* with clear failure definitions, we can claim more precise metrics/diagnostics than broad survey discussions.

## Where GALILEO is weaker / needs to improve

- If reviewers care about broader “Embodied AI” context, we may need a short related-work paragraph that explains why we are *not* targeting physical simulators (scope control) while acknowledging this adjacent evaluation literature.

## Action items for GALILEO (experiments / method / writing)

- [ ] If we need one cross-domain related-work sentence: cite this survey as evidence that interactive/embodied tasks emphasize simulator fidelity + hardware tradeoffs, analogous to why our evaluation harness details matter.
- [ ] Optional: add a footnote/aside in related work clarifying scope: “we focus on text-based multi-round evaluation; embodied evaluation uses physics simulators and has its own sim-to-real concerns.”

## Quotes / details to potentially cite

- Abstract framing (sim-to-real gap; simulators narrow gap via physics modeling + photorealistic rendering):
  - “...sim-to-real transfer has emerged as a key approach, yet the sim-to-real gap persists.”
  - “This survey examines how physics simulators address this gap...”
- Survey’s navigation decomposition + explicit statement of two gaps (visual vs physics) from Section 2 discussion.
