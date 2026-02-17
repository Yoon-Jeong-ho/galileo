# HeroBench: A Benchmark for Long-Horizon Planning and Structured Reasoning in Virtual Worlds

- Year: 2025
- Venue: arXiv
- Authors: Petr Anokhin; Roman Khalikov; Stefan Rebrikov; Viktor Volkov; Artyom Sorokin; Vincent Bissonnette
- URL: https://arxiv.org/abs/2508.12782
- BibTeX key (if we add it): herobench2025
- Tags: agents, planning, long-horizon, virtual-worlds, benchmark

## One-sentence takeaway

HeroBench evaluates long-horizon, dependency-heavy planning by asking LLM agents to plan and execute multi-step crafting/combat objectives inside a controlled RPG-style virtual world with an executable simulator.

## What problem does it solve?

- Current “planning” evaluations for LLMs often rely on abstract / low-dimensional tasks (e.g., classic symbolic planning setups) that miss the layered dependencies of realistic long-horizon objectives.
- Existing game-like environments can be hard to interface with, require many low-level actions, and may leak mechanics via pretraining exposure.
- Need a benchmark where (i) tasks require explicit multi-step plans, (ii) correctness can be checked via execution, and (iii) difficulty can be scaled in a principled way.

## What is the core method / protocol?

- Environment: a discrete, grid-based RPG world (70 locations) containing:
  - resource nodes / workshops / monster spawns
  - 25 monsters, 17 resource types, and 208 items (gear + components)
- Tasks: two main types
  - Crafting-only tasks: craft a target item.
  - Combat tasks: defeat a target monster (often requires crafting prerequisite gear first).
- Key design choice for combat tasks: require the agent to *compute an (approximately) minimal winning gear set* before gathering/crafting.
  - Combat is turn-based and depends on HP, raw damage, elemental damage/resistance (fire/earth/water/air), and % amplifications.
  - Task generation includes a combat simulator plus a search procedure over candidate gear items (bounded by the monster’s level requirement) to identify a minimal winning set.
- Evaluation: run the agent’s plan in a deterministic simulator (paper notes a deterministic mode; the environment can also support stochastic drops).

## What are the key metrics?

- Task success rate (did the agent craft the item / defeat the monster under the environment’s validator).
- Breakdown/error analysis (qualitative categories): planning failures vs execution/step errors vs calculation/gear-selection errors.
- Difficulty-stratified performance (tasks vary by number of prerequisite items/steps).

## What are the main results?

- Across 25 evaluated LLMs (open + closed) and two agentic architectures, performance varies widely—more than what is typically visible on standard math/programming reasoning benchmarks.
- Error analysis highlights recurring weaknesses in:
  - generating robust high-level plans that respect dependencies
  - reliably executing structured multi-step action sequences without derailment

(Details like exact numbers/tables were not captured in this rapid pass; revisit PDF if we need specific scores.)

## How is this similar to GALILEO?

- If GALILEO targets agent robustness in long-horizon/multi-step settings, HeroBench is a close “evaluation neighbor”: executable environment + explicit dependency chains + long-horizon plan/execute loop.
- Provides a concrete protocol for separating *planning* (high-level strategy + gear computation) from *execution* (resource gathering/crafting/combat actions).

## How is this different from GALILEO?

- HeroBench is a game-like virtual world benchmark focused on long-horizon planning/crafting/combat dependencies (not primarily on dialogue-driven multi-turn robustness).
- Emphasizes numerical/stat-based gear optimization as a required subproblem.
- Uses an environment-specific simulator/validator; transfer to other domains depends on how GALILEO is instantiated.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO is meant to be domain-agnostic, it may offer cleaner abstractions than an RPG-specific benchmark.
- If GALILEO focuses on robustness metrics (e.g., drift/turn-of-failure), it could provide richer longitudinal diagnostics than pure success rates.

## Where GALILEO is weaker / needs to improve

- If GALILEO lacks an executable “world model” for validation, HeroBench’s simulator-backed evaluation is a strong point to match.
- If GALILEO does not explicitly enforce dependency-aware planning (e.g., minimal prerequisites), HeroBench’s task construction suggests a way to tighten difficulty control.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding a *simulator-backed* long-horizon track (or a toy environment) where plans can be executed and validated, not just judged.
- [ ] Add difficulty controls tied to dependency depth/width (number of prerequisite items/steps), similar to HeroBench’s construction.
- [ ] If relevant, adopt a “minimal winning plan/gear” notion to encourage resource-efficient planning and enable more fine-grained scoring.

## Quotes / details to potentially cite

- “We introduce HeroBench, a novel benchmark designed specifically to evaluate long-horizon planning and structured reasoning within complex RPG-inspired virtual worlds.”
- Environment scale (from the HTML version): 70 locations; 25 monsters; 17 resource types; 208 unique items.
- Task types: crafting-only vs defeat-enemy tasks; combat tasks require explicit optimal gear calculation over elemental stats/resistances.
