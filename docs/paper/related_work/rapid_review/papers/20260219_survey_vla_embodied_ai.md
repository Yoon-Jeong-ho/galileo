# A Survey on Vision-Language-Action Models for Embodied AI

- Year: 2025 (arXiv v5, posted 2024-05; use paper’s stated year in bib as needed)
- Venue: arXiv (survey)
- Authors: Yueen Ma; Zixing Song; Yuzheng Zhuang; Jianye Hao; Irwin King
- URL: https://arxiv.org/html/2405.14093v5
- BibTeX key (if we add it): Ma2025SurveyVLA (suggested)
- Tags: survey, vision-language-action, embodied-ai, robotics

## One-sentence takeaway

A taxonomy-style survey that organizes recent vision-language-action (VLA) work into (i) component advances, (ii) low-level action policies, and (iii) high-level task planners, plus a resources/benchmarks overview.

## What problem does it solve?

- Helps researchers navigate a fast-moving space of “VLA” models (vision+language → actions) for embodied/robotic tasks by providing definitions, categorization, and pointers to datasets/benchmarks.

## What is the core method / protocol?

- Survey + taxonomy organized around a common hierarchical robotics framing:
  - **Components**: RL/sequence modeling, pretrained visual representations, dynamics learning, (visual/textual) world models, reasoning.
  - **Low-level control policies**: models that directly predict robot actions conditioned on visual observations + language.
  - **High-level task planners**: models that decompose long-horizon instructions into sub-tasks, often in a modular planner-controller setup.
- Summarizes representative papers and provides resource lists (datasets, simulators, benchmarks). Also links an “Awesome-VLA” project list.

## What are the key metrics?

- Not a single metric paper; metrics depend on the underlying benchmarks surveyed (manipulation/navigation success rates, task completion, generalization, etc.).

## What are the main results?

- Primary “result” is the organizational structure + synthesis:
  - A broadened definition of VLA (beyond only LLM-based “Large VLAs”) as any model mapping vision+language to actions.
  - A taxonomy aligned with typical hierarchical robot system design (planner → controller).
  - A comparative discussion of representation learning objectives (CLIP-like contrastive; MAE; time-contrastive video objectives; self-distillation) and their relevance to robotics.

## How is this similar to GALILEO?

- If GALILEO is positioning itself as a **foundation model** with a clear taxonomy of neighboring work + benchmarks, this survey is an example of:
  - crisp definitions and categorization;
  - “resources/benchmarks” as first-class related-work structure.

## How is this different from GALILEO?

- This is **robotics embodied AI** (vision-language-action), not geospatial/EO.
- No new model/algorithmic contribution to benchmark against; mainly a map of the space.

## Where GALILEO is stronger / cleaner (if true)

- GALILEO (as a concrete method paper) can provide:
  - a single coherent objective/training recipe;
  - ablations and direct benchmark deltas.

## Where GALILEO is weaker / needs to improve

- If GALILEO’s related work is sprawling, borrowing this survey’s “taxonomy + resources” structure could improve narrative clarity.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding a short “taxonomy of neighbors + resources” subsection (even 1 figure/table) to make the related-work section more navigable.

## Quotes / details to potentially cite

- Definition framing: VLAs process vision+language and “generate actions” for language-conditioned robotic tasks; survey organizes work into components / control policies / task planners.
- Project list: https://github.com/yueen-ma/Awesome-VLA
