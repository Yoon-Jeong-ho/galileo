# Perspective Dial: Measuring Perspective of Text and Guiding LLM Outputs

- Year: 2025
- Venue: arXiv (cs.CL, cs.AI)
- Authors: Taejin Kim; Siun-Chuon Mau; Konrad Vesey
- URL: https://arxiv.org/abs/2506.23377
- BibTeX key (if we add it): perspective-dial-kim-2025
- Tags: perspective, bias, measurement, controllability, prompt-engineering, contrastive-learning

## One-sentence takeaway

Proposes an empirical “Perspective Space” metric (learned via contrastive/siamese encoder) and a feedback loop that uses greedy coordinate-descent prompt edits to steer an LLM’s outputs toward a target measured perspective.

## What problem does it solve?

- Lack of a *quantitative* way to measure “perspective/viewpoint” (related to bias) of arbitrary text and, crucially, a way to *control* the perspective of LLM outputs using that measurement.
- Practical need: detect/track/mitigate unwanted bias/perspective shifts in mission-critical LLM deployments.

## What is the core method / protocol?

- **Perspective Space (measurement):**
  - Define “perspective” empirically using example texts representing contrasting viewpoints on the same topic.
  - Learn a metric space inherited from an embedding space, trained with **contrastive learning** using a **BERT-based siamese architecture** (per intro).
  - Use distances/positions in this space as a scalar/metric feedback signal for perspective.
- **Systematic Prompt Engineering (control):**
  - Iteratively edit the *user prompt* to push generated outputs toward a desired region/target in Perspective Space.
  - Uses a **greedy coordinate descent** style procedure driven by measurement feedback (evaluate output → measure perspective → modify prompt → repeat).
  - Notably decouples the “measurer” LM from the “generator” LLM (measurement model can differ from the LLM being steered).

## What are the key metrics?

- Perspective measurement in the learned **Perspective Space** (distance / similarity relative to contrast sets or targets).
- For control: reduction in distance to a target perspective (or movement along the relevant direction) over prompt-optimization iterations.

## What are the main results?

- Demonstrates feasibility of (a) training a perspective metric with limited labeled contrast data and (b) using that metric as a feedback signal to steer LLM outputs via iterative prompt optimization.
- Claims practical deployability since the control loop operates at prompting time (no model weight changes needed for the generator).

## How is this similar to GALILEO?

- Fits a “**measure → steer**” paradigm: define a quantitative evaluation signal for a latent property of text, then optimize generation to hit a target.
- Relevant framing for any GALILEO components that separate *scoring/evaluation* from *generation* (e.g., external critics, reward/metric models, or constraint checkers).

## How is this different from GALILEO?

- Focus is specifically on **perspective/viewpoint** as an empirically learned metric space, rather than (e.g.) factuality, uncertainty, calibration, provenance, or task performance.
- Uses **prompt-only** optimization (greedy coordinate descent over prompt features) rather than training-time alignment, fine-tuning, or RL-style updates.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO’s targets/constraints are more formally defined (or decomposable into verifiable checks), it may offer clearer guarantees than an empirically defined “perspective” metric.

## Where GALILEO is weaker / needs to improve

- If GALILEO lacks an explicit “dial”/control interface for nuanced discourse properties (viewpoint, framing), this work suggests a concrete way to operationalize such controls.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding a **learned metric space** for a discourse attribute (perspective/framing) as an auxiliary evaluation signal.
- [ ] Prototype a lightweight **prompt-optimization loop** (greedy coordinate descent) using an external evaluator model as the feedback function.
- [ ] In related work, cite as an example of **measurement-driven controllability** without modifying generator weights.

## Quotes / details to potentially cite

- “Perspective-Dial consists of two main components: a (1) metric space, dubbed Perspective Space, that enables quantitative measurements of different perspectives regarding a topic, and the use of (2) Systematic Prompt Engineering that utilizes greedy-coordinate descent to control LLM output perspective based on measurement feedback from the Perspective Space.” (abstract)
- Claims perspective control is “agnostic” to the Perspective Space model because the LLM backing the Perspective Space can be unrelated to the LLM generating output (intro).
