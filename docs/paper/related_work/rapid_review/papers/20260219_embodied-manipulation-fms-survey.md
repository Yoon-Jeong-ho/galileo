# Embodied Robot Manipulation in the Era of Foundation Models: Planning and Learning Perspectives

- Year: 2025
- Venue: arXiv (cs.RO) survey
- Authors: Shuanghao Bai; Wenxuan Song; Jiayi Chen; Yuheng Ji; Zhide Zhong; Jin Yang; Han Zhao; Wanqi Zhou; Zhe Li; Pengxiang Ding; Cheng Chi; Chang Xu; Xiaolong Zheng; Donglin Wang; Haoang Li; Shanghang Zhang; Badong Chen
- URL: https://arxiv.org/abs/2512.22983
- BibTeX key (if we add it): bai2025embodied
- Tags: robotics, manipulation, foundation models, planning, control, survey

## One-sentence takeaway

A recent survey that frames foundation-model-era manipulation as a high-level planning problem (reasoning over language/code/motion/affordances/3D) plus a low-level learning-based control problem, and summarizes taxonomies + open challenges.

## What problem does it solve?

- Provides an organizing abstraction for a fast-moving area (robotic foundation models for manipulation) where papers span different levels (task planning vs control) and modalities (VLMs, 3D, code, etc.).
- Aims to clarify the design space and common components for long-horizon manipulation stacks.

## What is the core method / protocol?

- Survey + taxonomy.
- Unifies learning-based manipulation methods into:
  - High-level planning: extends classical task planning to reasoning over language, code, motion, affordances, and 3D representations.
  - Low-level control: taxonomy organized by training paradigm, with axes including input modeling, latent representation learning, and policy learning.
- Discusses challenges/directions: scalability, data efficiency, multimodal physical interaction, safety.

## What are the key metrics?

- Not a single benchmark paper; metrics depend on the underlying methods reviewed.
- Common implicit axes the survey is about:
  - Success rate / task completion in long-horizon manipulation.
  - Generalization across objects/tasks/environments.
  - Data efficiency (demonstrations, interaction steps).
  - Safety / constraint satisfaction during execution.

## What are the main results?

- A structured decomposition (planning vs control) and taxonomies that can be used to position new GALILEO-related work.
- A curated set of representative studies (the authors note this is a “re-architected core” derived from a longer survey at arXiv:2510.10903).

## How is this similar to GALILEO?

- If GALILEO is positioned as an embodied/manipulation method (or a system) that uses foundation models, this survey’s abstraction (high-level planner + low-level controller) is a clean way to explain GALILEO.
- The emphasis on long-horizon structured decision making (reasoning over language/affordances/3D/motion) aligns with typical GALILEO-style narratives.

## How is this different from GALILEO?

- This is a survey, not a new algorithm or empirical contribution.
- It focuses on organizing prior work rather than proposing a concrete architecture, training recipe, or benchmark evaluation.

## Where GALILEO is stronger / cleaner (if true)

- A single, end-to-end method description and a focused evaluation can be clearer than survey-style categorization.
- If GALILEO has a principled interface between planner/controller (or better ablations), that can go beyond the survey’s conceptual decomposition.

## Where GALILEO is weaker / needs to improve

- If our writing does not explicitly separate (1) planning representations/reasoning and (2) control learning paradigm, the survey will make our framing look less systematic.
- If we do not address data-efficiency and safety claims explicitly, this survey highlights them as key open issues.

## Action items for GALILEO (experiments / method / writing)

- [ ] In related work, explicitly organize comparisons under two headings: (A) high-level planning/reasoning representations and (B) low-level control learning paradigms.
- [ ] Add a brief “open challenges” paragraph (scalability, data efficiency, multimodal physical interaction, safety) and state which ones GALILEO tackles.
- [ ] Consider adding an ablation table aligned to the survey axes (input modality choice; latent representation; policy learning objective).

## Quotes / details to potentially cite

- “Organizes recent learning-based approaches within a unified abstraction of high-level planning and low-level control.”
- “Extend[s] the classical notion of task planning to include reasoning over language, code, motion, affordances, and 3D representations.”
- “Identify open challenges … scalability, data efficiency, multimodal physical interaction, and safety.”
