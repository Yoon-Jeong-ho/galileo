# A Survey on Vision-Language-Action Models for Embodied AI

- Year: 2024
- Venue: arXiv (survey)
- Authors: Yueen Ma et al.
- URL: https://arxiv.org/abs/2405.14093
- BibTeX key (if we add it): Ma2024SurveyVLA
- Tags: vla, embodied-ai, survey

## One-sentence takeaway

A broad survey that taxonomizes vision-language-action models (VLAs) for embodied AI into (1) component advances, (2) low-level action-predicting control policies, and (3) high-level task planners, plus datasets/benchmarks and open challenges.

## What problem does it solve?

- Rapidly growing, fragmented VLA literature (robotics + VLM/LLM + control) needs a unified taxonomy and an index of resources (datasets, simulators, benchmarks) to make the space navigable.

## What is the core method / protocol?

- Survey + taxonomy of VLAs organized into three major research lines:
  - Components of VLAs (e.g., pretrained visual reps, video reps, dynamics/world models, reasoning, policy steering).
  - Low-level control policies that predict low-level actions (incl. transformer policies, diffusion-based policies, 3D vision, motion planning, “large VLA”).
  - High-level task planners that decompose long-horizon tasks into subtasks (monolithic vs modular; language-based vs code-based).
- Provides curated resources (datasets/benchmarks) and a companion “Awesome-VLA” repository.

## What are the key metrics?

- Not a single-metric paper; it points to task-level embodied/robotics benchmarks (real-world robot datasets, simulation benchmarks, task-planning benchmarks, embodied QA benchmarks).

## What are the main results?

- A structured map of the field and consolidated pointers to resources.
- Highlights challenges/future directions including (per ToC): safety, generalization, multimodality, long-horizon task frameworks, real-time responsiveness, multi-agent systems, ethical/societal implications, applications.

## How is this similar to GALILEO?

- Both are concerned with agentic systems executing multi-step tasks; the “task planner” line (decomposition, instruction following, long-horizon control) rhymes with GALILEO’s focus on multi-turn robustness and stability across rounds.
- The “safety first” / robustness framing in embodied settings can be a useful analogy when positioning GALILEO’s robustness contributions.

## How is this different from GALILEO?

- Domain mismatch: this is embodied robotics (vision-language-action control + planning), whereas GALILEO targets multi-turn conversational/decision robustness phenomena (drift, persuasion/sycophancy, belief revision controls).
- Methodology mismatch: survey/taxonomy vs new algorithm/benchmark contributions.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO has controlled multi-turn stress tests and explicit drift/persuasion measurements, it likely offers clearer causal evaluation than many embodied benchmarks that conflate perception/control/sim-to-real.

## Where GALILEO is weaker / needs to improve

- If GALILEO aims to connect to “general agents,” it may be weaker on grounded action execution and long-horizon planning in partially observable environments (a core concern in VLAs).

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider borrowing vocabulary/structure from VLA surveys when writing GALILEO related-work: separate “components,” “low-level policy,” and “high-level planner” analogs for LLM agents (e.g., representation/elicitation, action policy, planner/controller).
- [ ] If we want a broader AGI positioning, add a short paragraph bridging conversational multi-turn robustness to long-horizon task planning (embodied or tool-using).

## Quotes / details to potentially cite

- Abstract framing: VLAs are “multimodal models … referred to as vision-language-action models (VLAs) … generate actions,” and the survey organizes work into components, low-level control policies, and high-level task planners; plus “datasets, simulators, and benchmarks,” and discusses challenges/future directions.
