# Embodied AI with Foundation Models for Mobile Service Robots: A Systematic Review

- Year: 2025
- Venue: arXiv (systematic review; v2 Jan 2026, resubmitted to Robotics)
- Authors: Matthew Lisondra (and coauthors; see PDF for full list)
- URL: https://arxiv.org/abs/2505.20503
- BibTeX key (if we add it): lisondra2025embodied
- Tags: robotics, mobile, service-robots, foundation-models, systematic-review, embodied-ai

## One-sentence takeaway

A systematic review of how foundation models (LLMs/VLMs/MLLMs/VLAs) are being integrated into mobile service robots, organized around language-to-action, multimodal perception, uncertainty/safety, and real-time deployment constraints.

## What problem does it solve?

- Consolidates and categorizes rapidly growing literature on foundation-model-driven embodied AI specifically for *mobile service robots* (domestic, healthcare, service automation), highlighting recurring technical gaps and non-technical (ethics/HRI) concerns.

## What is the core method / protocol?

- Systematic review (survey) rather than a new algorithm.
- Frames progress via several recurring capability “blocks”:
  - Language-conditioned control / instruction-to-action mapping
  - Multimodal sensor fusion and perception in human-centered spaces
  - Uncertainty-aware reasoning / safe decision making
  - Efficient scaling / resource-constrained, real-time onboard deployment
- Discusses application domains and broader implications (privacy, governance, human-in-the-loop).

## What are the key metrics?

- Not a primary-evaluation paper; metrics are those commonly used across reviewed works.
- Useful to mine for: task success rate in long-horizon mobile manipulation, navigation success, planning reliability, latency / compute / memory footprint, safety incident rates (or proxy measures), and HRI/user satisfaction where applicable.

## What are the main results?

- Claims “first systematic review” focused on foundation models in *mobile service robotics*.
- Synthesizes challenges: translating NL instructions into executable actions, robust multimodal perception, uncertainty estimation for safety, and computational constraints for onboard real-time inference.
- Identifies future directions: reliability & lifelong adaptation, privacy-aware + resource-constrained deployment, and governance/human-in-the-loop frameworks.

## How is this similar to GALILEO?

- Overlaps in framing: language-to-action grounding, multimodal perception, safety/uncertainty, and deployment constraints are exactly the pain points GALILEO-style systems must address.
- Provides a “map” of adjacent approaches and terminology that can help position GALILEO.

## How is this different from GALILEO?

- Survey paper (taxonomy + synthesis), not a concrete method or experimental system.
- Emphasis is mobile *service robots* broadly (navigation + interaction + assistance), so it may cover a wider set of tasks and platforms than GALILEO’s scope.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO has a crisp, end-to-end reproducible pipeline and ablations, it can appear stronger than many reviewed works that are demo-driven or evaluation-light.

## Where GALILEO is weaker / needs to improve

- If GALILEO lacks explicit uncertainty estimation / safety mechanisms and/or realistic onboard compute profiling, this review’s framing suggests reviewers will expect that discussion.

## Action items for GALILEO (experiments / method / writing)

- [ ] Use this review’s challenge framing to structure the related-work section (subsections aligned to: instruction→action, multimodal perception, uncertainty/safety, deployment/efficiency).
- [ ] Add a short “deployment constraints” paragraph in GALILEO: latency, memory, and what runs on-board vs off-board.
- [ ] Explicitly discuss uncertainty/safety: how failures are detected, how the system falls back, and what is guaranteed/not guaranteed.

## Quotes / details to potentially cite

- “Despite this progress, embodied AI for mobile service robots continues to face fundamental challenges related to the translation of natural language instructions into executable robot actions, multimodal perception in human-centered environments, uncertainty estimation for safe decision-making, and computational constraints for real-time onboard deployment.”
- Future directions called out: “reliability and lifelong adaptation, privacy-aware and resource-constrained deployment, and governance and human-in-the-loop frameworks…”
