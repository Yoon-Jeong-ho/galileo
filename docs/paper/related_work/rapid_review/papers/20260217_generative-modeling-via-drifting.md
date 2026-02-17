# Generative Modeling via Drifting

- Year: 2026
- Venue: arXiv
- Authors: Mingyang Deng; He Li; Tianhong Li; Yilun Du; Kaiming He
- URL: https://arxiv.org/abs/2602.04770
- BibTeX key (if we add it): Deng2026Drifting
- Tags: drifting, one-step generation, pushforward, distribution-matching, contrastive, kernels

## One-sentence takeaway

A one-step generator is trained by explicitly modeling **how the generator’s pushforward distribution drifts during optimization**, using a “drifting field” loss that goes to zero at distributional equilibrium.

## What problem does it solve?

- Single-step (1-NFE) generators are attractive for speed, but many high-quality approaches rely on multi-step diffusion/flow inference or distillation from multi-step teachers.
- They aim to train a **high-quality one-step** generative model *from scratch* without SDE/ODE inference-time dynamics.

## What is the core method / protocol?

- View a generator as a pushforward map: sample \(\epsilon \sim p(\epsilon)\), output \(x=f(\epsilon)\), inducing \(q = f_\# p(\epsilon)\).
- During training, optimization produces a sequence \(\{f_i\}\) and thus a sequence of pushforward distributions \(\{q_i\}\).
- Introduce a **drifting field** (a vector field governing sample movement) that depends on the current generated distribution and the data distribution, and is defined so it becomes **zero at equilibrium** (when \(q \approx p_{data}\)).
- Train by minimizing a loss derived from this drifting field, so that SGD updates “evolve” the pushforward distribution toward the data distribution.
- The paper relates the drifting field construction to kernel / positive-vs-negative sampling ideas (contrastive-flavored framing).

## What are the key metrics?

- Image generation quality via **FID** (reported both latent-space and pixel-space protocols on ImageNet 256×256).

## What are the main results?

- ImageNet 256×256, **1-step (1-NFE)** generation:
  - FID **1.54** (latent-space generation protocol)
  - FID **1.61** (pixel-space generation protocol)
- Positioned as state-of-the-art among one-step methods, and competitive with some multi-step diffusion/flow approaches.

## How is this similar to GALILEO?

- Conceptual overlap in the language of **drift** and **equilibrium**:
  - They formalize training as an iterative process that induces a *trajectory* over distributions, and define an equilibrium condition.
- Methodologically, it is another example of turning a qualitative “drift toward a target state” intuition into a quantitative objective.

## How is this different from GALILEO?

- Different domain and object of study: image generation / distribution matching, not multi-turn conversational behavior or belief drift.
- Their “drift” is a **training-time distributional evolution** of a generator; GALILEO’s drift concerns **interaction-time behavioral/stance evolution** under pressure vs evidence.
- Metrics are generative-quality (FID), not time-to-failure / flip dynamics / recovery.

## Where GALILEO is stronger / cleaner (if true)

- GALILEO’s framing (as intended) explicitly distinguishes **pressure-driven drift vs evidence-driven revision** and emphasizes **trajectory-level evaluation** (time-to-event, recovery), which this work does not target.

## Where GALILEO is weaker / needs to improve

- If GALILEO uses “drift/equilibrium” language informally, this paper is a reminder that reviewers may expect clearer mathematical objects (state, field/dynamics, equilibrium condition) and an explicit “goes to zero at equilibrium” style argument.

## Action items for GALILEO (experiments / method / writing)

- [ ] If we use *equilibrium* language, add a crisp definition (what state? what metric?) and explicitly state what it means for “drift to be zero”.
- [ ] Consider whether any of our drift metrics can be described as (or derived from) a **field/gradient** that vanishes at a desired stable point (even if purely conceptual).

## Quotes / details to potentially cite

- “We propose a new paradigm called Drifting Models, which evolve the pushforward distribution during training and naturally admit one-step inference.”
- “We introduce a drifting field that governs the sample movement and achieves equilibrium when the distributions match.”
- Reported ImageNet 256×256 FID: 1.54 (latent) / 1.61 (pixel), both at 1-NFE.
