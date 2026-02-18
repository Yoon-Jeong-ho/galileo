# A Comprehensive Survey on World Models for Embodied AI

- Year: 2025
- Venue: arXiv (cs.CV)
- Authors: Xinqing Li; Xin He; Le Zhang; Min Wu; Xiaoli Li; Yun Liu
- URL: https://arxiv.org/abs/2510.16732
- BibTeX key (if we add it): Li2025WorldModelsSurvey
- Tags: world-models, embodied-ai, survey, taxonomy, evaluation

## One-sentence takeaway

A broad survey that unifies “world model” work for embodied AI via a 3-axis taxonomy (functionality × temporal modeling × spatial representation) and highlights evaluation gaps (physical consistency, long-horizon coherence, and real-time efficiency).

## What problem does it solve?

- The literature uses inconsistent definitions/taxonomies for world models across robotics, driving, and video generation.
- Evaluation is fragmented: metrics often over-emphasize pixel fidelity rather than physical/temporal consistency and downstream control utility.

## What is the core method / protocol?

- Survey + unifying conceptual framework:
  - **Functionality:** Decision-coupled (task-specific, model-based RL / control) vs **general-purpose** (task-agnostic prediction/simulation).
  - **Temporal modeling:** **Sequential simulation/inference** (autoregressive rollout; filtering/posterior style, Dreamer-like) vs **global difference prediction** (predict future states/changes in parallel; includes masked/JEPA-style).
  - **Spatial representation:** global latent vector; token feature sequence; spatial latent grid (e.g., BEV/voxels); decomposed rendering representations (e.g., NeRF / 3D Gaussian Splatting).
- Background formalization in a POMDP with learned latent state z_t, dynamics prior p(z_t|z_{t-1}, a_{t-1}), filtered posterior q(z_t|z_{t-1}, a_{t-1}, o_t), and reconstruction p(o_t|z_t), trained via ELBO.
- Organizes datasets/metrics across robotics, autonomous driving, and video; provides comparative tables of representative models.

## What are the key metrics?

- Surveyed metrics categories (not a single new metric):
  - Pixel-level prediction quality (image/video metrics).
  - State-level understanding/consistency.
  - Downstream task performance (control success / planning performance).
- Emphasized need for metrics targeting **physical consistency** and **long-horizon temporal coherence**, not just perceptual similarity.

## What are the main results?

- Main “result” is a structured map of the space and a synthesis of challenges:
  - Dataset fragmentation + lack of unified benchmarks.
  - Long-horizon rollout consistency and error accumulation remain core obstacles.
  - Tension between model quality and **computational efficiency** needed for real-time control.
  - Spatial representations with stronger geometric bias can better support occlusion/object permanence and geometry-aware planning.
- Provides extensive tables of methods (robotics + driving) mapped into the taxonomy.

## How is this similar to GALILEO?

- If GALILEO targets predictive internal simulation for embodied agents, it fits directly into the “world model” framing.
- Highlights the same design axes GALILEO must decide on:
  - Decision-coupled vs general-purpose modeling goal.
  - Sequential rollout vs more global prediction.
  - Choice of spatial representation (tokens vs grids vs explicit 3D/decomposed rendering).

## How is this different from GALILEO?

- This is a **survey/taxonomy** paper; it does not propose a single new model or training recipe.
- Emphasis is on organizing prior work and calling out evaluation issues, rather than delivering a specific system.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO has a crisp, task-relevant evaluation (closed-loop control or planning utility) and clear compute/latency reporting, that would address the survey’s critique of pixel-centric evaluation.

## Where GALILEO is weaker / needs to improve

- Ensure GALILEO’s evaluation demonstrates **physical consistency** and **long-horizon** behavior, not just short-horizon reconstructions.
- If GALILEO relies on purely token/latent predictions, consider whether geometry-aware representations are needed for occlusion and object permanence.

## Action items for GALILEO (experiments / method / writing)

- [ ] In the related-work section, explicitly position GALILEO in a 3-axis taxonomy (functionality/temporal/spatial) to reduce ambiguity for readers.
- [ ] Add (or justify absence of) metrics for physical consistency and long-horizon coherence; separate perceptual quality from controllability.
- [ ] Report compute/latency and discuss the quality–efficiency trade-off for real-time control.

## Quotes / details to potentially cite

- “World models serve as internal simulators that capture environment dynamics, enabling forward and counterfactual rollouts…” (Abstract)
- Open challenges emphasized: unified datasets/benchmarks; metrics assessing physical consistency over pixel fidelity; real-time efficiency trade-offs; long-horizon temporal consistency and error accumulation. (Abstract/Intro)
- Three-axis taxonomy: functionality (decision-coupled vs general-purpose), temporal modeling (sequential simulation/inference vs global difference prediction), spatial representation (global latent vector / token sequence / spatial grid / decomposed rendering). (Intro/Taxonomy)
