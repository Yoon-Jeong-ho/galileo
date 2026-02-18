# Foundation Model Driven Robotics: A Comprehensive Review

- Year: 2025
- Venue: arXiv (cs.RO)
- Authors: Ammar Waheed; Muhammad Tayyab Khan; (others not captured in quick scrape)
- URL: https://arxiv.org/abs/2507.10087
- BibTeX key (if we add it): waheed2025foundationmodeldrivenrobotics
- Tags: robotics, foundation-models, survey, llm, vlm, sim2real, safety

## One-sentence takeaway

A structured survey of how LLMs/VLMs are being integrated into end-to-end robotic systems (perception→planning→control), organized by deployment setting (simulation/open-world/sim-to-real/adaptation) and emphasizing real-world feasibility bottlenecks (grounding, safety, compute, data).

## What problem does it solve?

- Fragmentation in prior surveys: many reviews cover isolated capabilities (e.g., language planning or vision grounding) but not system-level integration and practical feasibility.
- Need for a coherent taxonomy that connects “semantic reasoning” capabilities of foundation models to embodied constraints (real-time control, sensing, safety).

## What is the core method / protocol?

- Survey / taxonomy paper (not a new model).
- Organizes recent FM-in-robotics work into application categories highlighted in the abstract:
  - simulation-driven design
  - open-world execution
  - sim-to-real transfer
  - adaptable robotics
- Discusses enabling trends (procedural scene generation, policy generalization, multimodal reasoning) and recurrent bottlenecks:
  - limited embodiment / weak grounding from language priors to physical state
  - lack of high-quality multimodal robot data
  - safety risks in open-world execution
  - computational constraints for on-robot inference + real-time responsiveness

## What are the key metrics?

- Not applicable as a single benchmarked method; the paper discusses evaluation themes across the literature.
- Implicit evaluation axes repeatedly emphasized:
  - generalization to novel tasks/environments (open-world robustness)
  - sim-to-real transfer quality
  - real-time performance/latency constraints
  - safety/reliability/trustworthiness in deployment

## What are the main results?

- Provides a “system-level” synthesis and a roadmap framing the gap between semantic reasoning and physical intelligence.
- Highlights that LLM-only reasoning is insufficient without perception/feedback/grounding mechanisms; integration remains the hard part.

## How is this similar to GALILEO?

- Shares a “practical feasibility” lens: emphasizes deployment constraints (safety, reliability, compute) rather than only capability demos.
- Useful framing language for discussing integrated pipelines (multiple components, failure modes, trust/resilience) instead of isolated modules.

## How is this different from GALILEO?

- Robotics-focused (embodiment, sim-to-real, perception/control), whereas GALILEO’s contribution is likely in model evaluation/monitoring methodology (depending on section of the paper).
- No new algorithmic proposal or benchmark; it is an organizing survey.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO provides a concrete protocol/metric suite and reproducible evaluation, it will be more actionable than a broad survey.

## Where GALILEO is weaker / needs to improve

- If GALILEO discusses “real-world robustness” abstractly, this survey can be cited to justify the importance of grounding, real-time operation, and safety/trust as first-class constraints.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider citing this in the related-work narrative to motivate integrated, system-level evaluation (esp. real-time, grounding, safety, trust).
- [ ] Add a short paragraph (or checklist) of “embodied constraints” (grounding, latency, compute, safety) as a template for discussing deployment feasibility of any method.

## Quotes / details to potentially cite

- Abstract framing: categorizes applications across “simulation-driven design, open-world execution, sim-to-real transfer, and adaptable robotics,” and calls out bottlenecks like “limited embodiment, lack of multimodal data, safety risks, and computational constraints,” with open challenges in “real-time operation, grounding, resilience, and trust.”
