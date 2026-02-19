# Large Model Empowered Embodied AI: A Survey on Decision-Making and Embodied Learning

- Year: 2025
- Venue: arXiv (survey)
- Authors: Wenlong Liang; Rui Zhou; Yang Ma; Bing Zhang; Songlin Li; Yijia Liao; Ping Kuang
- URL: https://arxiv.org/html/2508.10399v1
- BibTeX key (if we add it): liang2025large
- Tags: survey, embodied-ai, decision-making, vla, imitation-learning, reinforcement-learning, world-models

## One-sentence takeaway

A broad survey of how “large models” (LLMs/VLMs/VLA, etc.) are used in embodied AI, organizing work around hierarchical vs end-to-end decision-making, learning (IL/RL), and (notably) world models.

## What problem does it solve?

- Rapidly growing embodied-AI literature around large models is fragmented across planning, control, learning, and application papers.
- Prior surveys often focus on the large models themselves (LLM/VLM) or on a single component (planning, learning), and may miss newer VLA/end-to-end trends.
- This survey aims to provide a more “systems-level” taxonomy focused on *decision-making* and *embodied learning*, plus an explicit discussion of *world models*.

## What is the core method / protocol?

- Survey + taxonomy (not a new algorithm).
- Organizes embodied AI pipeline and where large models plug in:
  - Decision-making paradigms:
    - Hierarchical: large models for high-level planning, low-level execution support, and feedback/iteration.
    - End-to-end: VLA (vision-language-action) models and enhancements (perception, action generation, deployment efficiency).
  - Learning paradigms:
    - Imitation learning: large models to build policy/representation, leverage demos/videos.
    - Reinforcement learning: large models to shape reward design and policy construction.
  - World models:
    - Design methods and their role in improving decision-making and learning via “mental simulation” / trialing.
- Uses “horizontal” (compare approaches) and “vertical” (evolution over time) analysis.

## What are the key metrics?

- As a survey, it does not introduce a single benchmark/metric.
- Covered works likely report standard embodied metrics depending on domain (success rate, SPL, task completion, reward, generalization across tasks/environments), but this paper’s own contribution is categorization and synthesis.

## What are the main results?

- Clear framing that current embodied AI with large models clusters into:
  - Hierarchical LLM/VLM-in-the-loop stacks (planner + controller/tools) vs
  - End-to-end VLA-style policies.
- Argues world models should be considered a first-class component in large-model-empowered embodied intelligence (for planning/learning through simulation).
- Summarizes open challenges (generalization, scalability, seamless environment interaction) and future directions.

## How is this similar to GALILEO?

- If GALILEO targets generalizable embodied decision-making/learning, this survey is directly adjacent: it maps the design space (hierarchical vs end-to-end) and highlights recurring failure modes (generalization, closed-loop interaction, data/feedback efficiency).
- The world-model section is likely relevant if GALILEO uses or motivates any latent dynamics / simulation / rollouts for planning or policy improvement.

## How is this different from GALILEO?

- This is a survey/taxonomy paper rather than a method with concrete training/inference pipeline and experimental results.
- Breadth over depth: it will not provide the implementation-level details, ablations, or novel empirical findings GALILEO should deliver.

## Where GALILEO is stronger / cleaner (if true)

- A focused, well-specified algorithm + reproducible experiments can be stronger than a broad survey for making a crisp contribution.
- If GALILEO provides a unified framework that concretely bridges hierarchical planning and end-to-end policies (or integrates a world model tightly), it can be positioned as “closing” one of the gaps the survey identifies.

## Where GALILEO is weaker / needs to improve

- Surveys often expose coverage gaps; ensure GALILEO’s related work clearly positions against:
  - Hierarchical LLM-planner + low-level policy stacks
  - VLA end-to-end models (data, architecture, deployment constraints)
  - World model approaches (what is learned, how rollouts are used, compounding error handling)

## Action items for GALILEO (experiments / method / writing)

- [ ] Use the survey’s hierarchy (high-level planning / low-level execution / feedback) as a checklist: explicitly state where GALILEO sits and what modules are learned vs prompted vs optimized.
- [ ] Add a short “hierarchical vs end-to-end” comparison paragraph in related work and justify why GALILEO chooses its paradigm.
- [ ] If relevant, include a “world model” subsection clarifying whether GALILEO uses explicit/implicit world modeling and how it impacts decision-making/learning.

## Quotes / details to potentially cite

- Abstract-level scope statement (paraphrase): the survey focuses on large-model empowerment for (i) hierarchical and end-to-end decision-making, (ii) embodied learning via IL/RL, and (iii) world models as an integrated component.
- “For the first time, we integrate world models into the survey of embodied AI…” (claim in intro/abstract; verify wording against the PDF if you cite directly).
