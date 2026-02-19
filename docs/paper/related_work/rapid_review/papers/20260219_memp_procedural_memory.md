# Memp: Exploring Agent Procedural Memory

- Year: 2025
- Venue: arXiv
- Authors: Runnan Fang, Yuan Liang, Xiaobin Wang, Jialong Wu, Shuofei Qiao, Pengjun Xie, Fei Huang, Huajun Chen, Ningyu Zhang
- URL: https://arxiv.org/abs/2508.06433
- BibTeX key (if we add it): memp2025
- Tags: agents, procedural-memory, lifelong-learning, memory-update

## One-sentence takeaway

Memp proposes a learnable procedural-memory repository for LLM agents, distilled from past trajectories and continuously updated (corrected/pruned), improving success and efficiency on similar long-horizon tasks and even transferring from stronger to weaker models.

## What problem does it solve?

- LLM agents often rely on brittle, manually engineered “procedural knowledge” (prompt templates / hand-coded skills) or knowledge entangled in model weights that is expensive to update.
- For long-horizon tasks, failures due to environment/tool changes are costly because agents restart from scratch instead of reusing prior successful procedures.

## What is the core method / protocol?

- Treat **procedural memory** as a first-class object with an explicit lifecycle: **Build → Retrieve → Update**.
- Build: distill agent trajectories into two granularities:
  - **Fine-grained step-by-step instructions** (actionable procedure)
  - **Higher-level script-like abstractions** (more general “subroutine”)
- Retrieve: explore alternative key/index construction strategies (described at a high level as variants such as query-vector matching vs keyword-vector matching) to fetch relevant procedures for a new task.
- Update: introduce multiple update policies for maintaining a living memory repository, including:
  - ordinary addition
  - validation filtering
  - reflection-based correction
  - dynamic discarding / deprecation
- Evaluate by coupling retrieval at test time with ongoing memory refinement so the repository “evolves with experience.”

## What are the key metrics?

- Task success rate / accuracy on long-horizon benchmarks.
- Execution efficiency: steps (and the paper claims token consumption reductions as well).

## What are the main results?

- As procedural memory is refined, the agent’s **success rate increases** and **steps decrease** on analogous tasks.
- The update mechanisms enable **continual improvement** during interaction with the test environment.
- **Transfer/migration**: procedural memory built using a stronger model can still provide meaningful gains when used with a weaker model.

## How is this similar to GALILEO?

- Shares the theme of improving agent performance via **experience distillation** into reusable artifacts.
- Emphasizes **long-horizon task robustness** and learning from prior trajectories.
- Calls out a full lifecycle (build/retrieve/update) rather than a static prompt/skill library.

## How is this different from GALILEO?

- Framed specifically as **procedural memory** (procedures/scripts) rather than broader project-level mechanisms (e.g., planning, verification, or environment modeling).
- Focus appears to be on an explicit **repository + retrieval + update policy** rather than a single unified algorithmic pipeline.
- Evaluation targets TravelPlanner and ALFWorld; alignment with GALILEO benchmarks may differ.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO has stronger guarantees around evaluation rigor, ablation clarity, or integration with verification/constraints, it may present a “cleaner” end-to-end story than a broad strategy exploration.

## Where GALILEO is weaker / needs to improve

- If GALILEO lacks an explicit **memory update/deprecation policy** (beyond accumulation), Memp highlights the importance of correcting and pruning procedural knowledge over time.
- If GALILEO’s reusable skills are primarily manually engineered, Memp provides a concrete motivation to distill them automatically from trajectories.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a section explicitly distinguishing **procedural memory** from episodic/semantic memory and motivate why procedures need their own lifecycle.
- [ ] Consider an ablation grid mirroring **Build/Retrieve/Update** choices (even if simplified) to position GALILEO relative to this line of work.
- [ ] If applicable, test **memory transfer**: build procedures with a stronger model, execute with a weaker one, measure gains.
- [ ] Document a **pruning/deprecation strategy** for learned procedures to avoid memory bloat and stale instructions.

## Quotes / details to potentially cite

- “LLM-based agents … suffer from brittle procedural memory that is manually engineered or entangled in static parameters.” (abstract)
- The paper frames the key design axis as strategies for “Build, Retrieval, and Update” of procedural memory and emphasizes continual correction/deprecation (abstract/introduction).
