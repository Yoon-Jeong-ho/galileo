# Can large language models independently complete tasks? A dynamic evaluation framework for multi-turn task planning and completion

- Year: 2025
- Venue: Neurocomputing (Elsevier) (ScienceDirect)
- Authors: Jun Gao; Junlin Cui; Huijia Wu; Liuyu Xiang; Han Zhao; Xiangang Li; Meng Fang; Yaodong Yang; Zhaofeng He (as listed on the ScienceDirect abstract page)
- URL: https://www.sciencedirect.com/science/article/pii/S0925231225008070
- BibTeX key (if we add it): <tbd>
- Tags: multi-turn, evaluation, task-planning, task-completion, dynamic-eval

## One-sentence takeaway
Proposes DEF-MT, a dynamic framework that evaluates *both* multi-turn task planning and task completion by forcing models to plan-and-respond sequentially and by dynamically generating user intents, finding that weak sub-task planning limits end-to-end completion on MultiWOZ 2.2.

## What problem does it solve?
- Existing multi-turn dialogue benchmarks often treat LLMs primarily as “agents,” and/or evaluate planning and completion separately.
- The paper targets evaluation of whether an LLM can *independently* complete complex tasks in multi-turn settings, with joint assessment of planning quality and completion success.

## What is the core method / protocol?
- **DEF-MT (Dynamic Evaluation Framework for Multi-Turn task planning and completion):**
  - Guides the model to generate **planning and responses sequentially** (plan → respond across turns) to quantify planning capability.
  - Uses a **dynamic data generation** approach to simulate complex, realistic user intents in multi-turn dialogue.
- Evaluates **9 mainstream models** on **MultiWOZ 2.2** (task-oriented dialogue).

## What are the key metrics?
- Not fully specified in the abstract.
- From the description, metrics likely include:
  - Planning quality / sub-task planning performance (their defined DEF-MT planning score).
  - Task completion / success under the dynamic intent setup.
  - Potentially standard MultiWOZ task success metrics (e.g., success/inform) — **needs full text to confirm**.

## What are the main results?
- On MultiWOZ 2.2 across 9 models, **sub-task planning weaknesses** are identified as a primary bottleneck preventing models from completing complex tasks in multi-turn scenarios.
- The framework is positioned as a reference direction for optimizing multi-turn task-capable LLMs.

## How is this similar to GALILEO?
- Shares the theme of **multi-turn evaluation where failures emerge over time** (planning breakdowns accumulate across turns).
- Emphasizes **protocol design** (how to structure multi-turn interactions) rather than only single-turn scoring.

## How is this different from GALILEO?
- DEF-MT focuses on **task-oriented dialogue (MultiWOZ)** with planning/completion, not social pressure / persuasion / belief stability.
- The “failure mode” is **task non-completion due to planning deficits**, not “belief flip,” “sycophancy,” or “recovery after misleading input.”

## Where GALILEO is stronger / cleaner (if true)
- GALILEO-style setups can more directly isolate **social pressure / persuasion** and define **turn-to-failure / time-to-failure** style robustness metrics tailored to *belief stability*.

## Where GALILEO is weaker / needs to improve
- If GALILEO claims general multi-turn competence, it may need clearer connections to **task planning + completion** protocols (not just conversational stance/belief robustness).

## Action items for GALILEO (experiments / method / writing)
- [ ] Add a related-work paragraph framing: multi-turn robustness is not only “belief/stance stability” but also **plan fidelity** across turns; cite DEF-MT as a task-oriented neighbor.
- [ ] Consider an ablation where GALILEO-style interventions are tested on **task planning drift** (e.g., plan changes under pressure / distraction) as an additional robustness axis.

## Quotes / details to potentially cite
- Abstract (problem framing): existing benchmarks “evaluate the planning and completion capabilities of the models individually, rather than simultaneously.”
- Abstract (main claim): “experiments … indicate that the existing models’ sub-task planning capabilities hinder their ability to complete complex tasks.”
