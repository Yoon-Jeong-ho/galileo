# Towards Forceful Robotic Foundation Models: a Literature Survey

- Year: 2025
- Venue: arXiv (survey)
- Authors: William Xie, Nikolaus Correll
- URL: https://arxiv.org/abs/2504.11827
- BibTeX key (if we add it): Xie2025ForcefulRoboticFMsSurvey
- Tags: robotics, foundation-models, force, tactile, survey

## One-sentence takeaway

A focused survey of transformer/diffusion-era manipulation policies that incorporate force/torque or tactile sensing, arguing today’s “robot foundation models” largely omit force and that the field lacks consistent benchmarks and clarity on when force truly matters.

## What problem does it solve?

- Organizes and synthesizes scattered recent work on learning manipulation policies that use force/torque and/or tactile sensing (especially transformer/diffusion policy-learning methods).
- Clarifies (and partially questions) the premise that force sensing/control is essential for learned generalist manipulation, by noting many current IL tasks are not yet in regimes where force is decisive.

## What is the core method / protocol?

- Survey / taxonomy (not a new algorithm):
  - Background on human tactile/proprioception, robotic force control, and policy learning.
  - Reviews a set of works learning tactile/force-conditioned policies with transformer or diffusion architectures.
  - Proposes analysis lenses like (estimated) force magnitude vs task makespan, and breaks down works by sensing modality, data collection, action spaces, and representation learning.

## What are the key metrics?

- Not a single benchmark; the paper aggregates reported metrics across prior work.
- One organizing lens: approximate force magnitude bins (0.1–1N, 1–10N, >10N) vs task length time (short vs long makespan).
- Notes that many papers do not report force magnitudes; authors estimate orders of magnitude from descriptions/videos.

## What are the main results?

- Argues current large-scale robot “foundation models” largely rely on vision + position control, omitting force/torque input/output.
- Finds a fragmented landscape:
  - Across reviewed works, many distinct tasks and few shared benchmarks (peg-in-hole appears somewhat common).
  - Few papers report the actual force magnitudes involved; this limits principled comparisons.
- Highlights an empirical/interpretive claim: for many imitation-learned tasks, performance is not yet at a regime where force feedback is the differentiator; force/touch can often be inferred or handled implicitly (e.g., impedance control, compliance, mechanism design).

## How is this similar to GALILEO?

- High-level similarity only: it is a survey attempting to systematize a complex space and identify “missing measurements” (here: force magnitudes / common benchmarks), which is analogous in spirit to GALILEO-style work that emphasizes evaluation rigor and stress-testing.

## How is this different from GALILEO?

- Different domain: robotic manipulation (force/tactile sensing + control) rather than multi-turn LLM robustness (sycophancy/persuasion/belief revision/stability).
- Method type: literature survey and taxonomy, not an evaluation protocol or algorithm targeting conversational robustness.

## Where GALILEO is stronger / cleaner (if true)

- N/A substantively (domain mismatch). If anything, GALILEO’s emphasis on controlled evaluation design and explicit reporting could be a “lesson” for robotics benchmarks, but that’s metaphorical.

## Where GALILEO is weaker / needs to improve

- Not applicable.

## Action items for GALILEO (experiments / method / writing)

- [ ] No direct action item (paper is likely out-of-scope for GALILEO’s related-work section).
- [ ] Optional: if writing an “analogy” sentence somewhere about missing modalities/measurements, this paper could be cited as an example of the importance of reporting the right signals (force magnitudes / shared benchmarks). Only include if you want a cross-domain aside.

## Quotes / details to potentially cite

- Abstract-level claim (paraphrase-worthy): they “articulate when and why forces are needed” and highlight opportunities for touch-based robot foundation models.
- Key positioning claim (Intro/Abstract): current robot foundation models “focus exclusively on visual input and position control”; force and touch are critical for contact-rich manipulation, but “why and how forces should be employed during learning remains still unclear.”
- Analysis claim (Abstract): for many tasks (only a few like pouring, peg-in-hole, delicate objects), imitation learning performance is not yet at dynamics where force truly matters; force/touch can be inferred from many modalities and often controlled implicitly.
