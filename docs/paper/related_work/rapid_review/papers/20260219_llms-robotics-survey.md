# Large Language Models for Robotics: A Survey

- Year: 2025 (arXiv v2)
- Venue: arXiv (cs.RO, cs.AI)
- Authors: Wensheng Gan; Zezheng Huai; Lichao Sun; Hechang Chen; Yongheng Wang; Ning Liu; Philip S. Yu
- URL: https://arxiv.org/html/2311.07226v2 (abs: https://arxiv.org/abs/2311.07226)
- BibTeX key (if we add it): gan2025llm_robotics_survey
- Tags: survey, llm-robotics, embodied, perception, planning, control, interaction

## One-sentence takeaway

A broad survey that frames LLMs as “robot brains” and organizes LLM-for-robotics work across perception, decision-making/planning, control, interaction, and cross-module coordination, with a discussion of deployment/safety challenges.

## What problem does it solve?

- Provides a taxonomy + literature overview for the rapidly growing area of using LLMs (and multimodal derivatives like VLM/VLA/VLN and agents) in embodied robotics.
- Aims to clarify where LLMs help (natural language interfaces, high-level planning, coordination) and what breaks in practice (compute, hallucinations/safety, multi-turn consistency, platform constraints).

## What is the core method / protocol?

- Survey paper (no new algorithm):
  - Background on LLMs + derivatives relevant to robotics (VLM/VLA/VLN/agents).
  - Organizes robotics systems into modular pipelines (perception / decision-making / control / interaction) and discusses cross-module coordination.
  - Highlights applications and open challenges.

## What are the key metrics?

- Not a primary benchmark paper; evaluation is discussed at a high level.
- Mentions common robotics concerns implicitly (task success, robustness, safety, compute/latency) but does not propose a single unified metric suite.

## What are the main results?

- Main “result” is the synthesis:
  - LLMs can improve natural language understanding/generation for HRI, and can support high-level planning / task decomposition.
  - Practical challenges: resource demands on robot platforms, hallucination / unsafe outputs, and multi-turn dialogue issues (context understanding, dialogue consistency).

## How is this similar to GALILEO?

- If GALILEO is positioned as an LLM-driven/agentic system (esp. for planning/interaction), this survey is useful as a broad map of adjacent work and standard module decomposition language (perception → decision → control → interaction).
- Explicitly calls out multi-turn dialogue consistency and safety filtering/control mechanisms as key issues for robot-facing LLM deployments.

## How is this different from GALILEO?

- Survey only: no concrete experimental protocol, no specific benchmark contribution, and no tight “closest-neighbor” evaluation setup.
- Focuses on robotics/embodiment broadly, rather than a targeted problem setting (whatever GALILEO’s specific contribution is).

## Where GALILEO is stronger / cleaner (if true)

- A focused system/benchmark paper can contribute:
  - a crisp task definition,
  - controlled ablations,
  - reproducible evaluation protocol,
  - and quantitatively grounded claims (all things surveys typically cannot provide).

## Where GALILEO is weaker / needs to improve

- If GALILEO’s related-work section is narrow, it may miss some of the survey’s organizing vocabulary and breadth (e.g., where to situate itself across perception/planning/control/interaction and “cross-module coordination”).

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider citing for a “big picture” intro paragraph motivating LLMs in robotics + listing core module decomposition.
- [ ] If GALILEO involves multi-turn interaction, use this survey as a source to motivate “multi-turn consistency/context” as an acknowledged open challenge.
- [ ] Mine the paper’s referenced systems (e.g., PaLM-E / language-conditioned planners) to add *more specific* closest-neighbor citations (this note did not exhaustively extract them).

## Quotes / details to potentially cite

- Abstract framing: introduces “dexterity intelligence” and positions LLMs as enabling better interaction/collaboration with robots.
- Challenges called out in intro: (i) compute/data demands on robot platforms; (ii) inaccurate/unreasonable/harmful generations → need filtering/control; (iii) multi-turn dialogue/context understanding/consistency.
