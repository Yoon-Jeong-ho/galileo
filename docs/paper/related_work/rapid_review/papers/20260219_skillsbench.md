# SkillsBench: Benchmarking How Well Agent Skills Work Across Diverse Tasks

- Year: 2026
- Venue: arXiv
- Authors: Xiangyi Li, Wenbo Chen, Yimin Liu, Shenghan Zheng, Xiaokun Chen, Yifeng He, et al.
- URL: https://arxiv.org/abs/2602.12670
- BibTeX key (if we add it): skillsbench2026li
- Tags: agents, skills, tool-use, benchmark, trajectories, multi-turn

## One-sentence takeaway

SkillsBench is an execution-verified benchmark showing that curated, modular “Skills” (procedural packages) can materially improve agent success rates, while self-generated Skills usually do not—highlighting big variance across domains and non-trivial negative side-effects.

## What problem does it solve?

- “Agent Skills” (structured procedural knowledge packages used at inference time) are widely used in agent systems, but there was no standard benchmark to quantify whether Skills help, when they help, and what design principles make them effective.
- Existing agent benchmarks measure *raw* agent/model capability, not the marginal gain from adding Skills (and the possibility of negative deltas).

## What is the core method / protocol?

- Construct **SkillsBench**: 84–86 tasks (paper states 86 in abstract; intro describes 84 curated tasks) across **11 domains**.
- Each task includes an isolated environment (Docker), an oracle solution, and **deterministic verifiers** (execution-based pass/fail).
- Evaluate each task under three conditions:
  - **No Skills**
  - **Curated Skills** (human-provided Skill packages)
  - **Self-generated Skills** (agent/model generates its own Skill-like artifact)
- Run a large-scale evaluation over **7 agent-model configurations** and **7,308 trajectories** across commercial agent harnesses (paper mentions Claude Code, Codex CLI, Gemini CLI).
- Additional analyses: domain-by-domain deltas; task-level negatives; “focused” vs “comprehensive” Skills packaging.

## What are the key metrics?

- Primary: **pass rate / resolution rate** (deterministically verified).
- Reported as absolute improvement in **percentage points (pp)** when adding Skills.
- Secondary: distribution/variance across domains and tasks; count of tasks with negative delta.

## What are the main results?

- **Curated Skills** improve average pass rate by about **+16.2 pp** (headline number).
- Gains vary substantially by domain (example range in paper: roughly **+4.5 pp** in Software Engineering up to **+51.9 pp** in Healthcare).
- A non-trivial fraction of tasks show **negative deltas** (paper: **16 of 84** tasks).
- **Self-generated Skills** provide **no benefit on average**, suggesting models cannot reliably author the procedural knowledge they benefit from consuming.
- **Focused Skills (2–3 modules)** outperform comprehensive documentation-like Skills.
- Skills can enable **smaller models + Skills** to match **larger models without Skills** (suggesting Skills are a lever for capability/efficiency tradeoffs).

## How is this similar to GALILEO?

- Both emphasize **multi-step / multi-turn agent behavior** rather than single-shot QA.
- Both argue for **robust, reproducible evaluation** (SkillsBench uses deterministic verifiers and containerized tasks).
- Both surface that “helpful additions” (extra context, procedures, scaffolds) can have **heterogeneous effects**, including regressions.

## How is this different from GALILEO?

- SkillsBench is primarily about **inference-time procedural augmentation** (Skills) and quantifying its marginal benefit, not primarily about conversational drift/sycophancy/stance pressure.
- The evaluation is **execution-based task completion** across domains, rather than measuring *belief/stance dynamics* under adversarial dialogue.
- It directly compares **curated vs self-generated** procedural artifacts, which is a different axis than typical multi-turn robustness benchmarks.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO targets conversational robustness phenomena (e.g., drift, pressure, stance flipping), it likely offers **more direct construct validity** for those dialogue-specific risks than a broad task benchmark.
- GALILEO may provide **more fine-grained behavioral metrics** (turn-level dynamics) beyond binary task success.

## Where GALILEO is weaker / needs to improve

- If GALILEO includes any “helper context” (policies, recipes, prompts), SkillsBench suggests we should:
  - quantify **marginal gains** of each scaffold,
  - and explicitly measure **negative deltas** (cases where scaffolds hurt).
- Consider adding a “scaffold ablation” suite: no scaffold vs curated scaffold vs model-generated scaffold.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add an explicit related-work paragraph positioning “Skills” as a form of **procedural context** and clarifying whether GALILEO’s interventions are closer to Skills vs prompts vs RAG.
- [ ] If GALILEO uses any procedural guidance, run **three-way ablation** similar to SkillsBench: none vs curated vs self-generated.
- [ ] Adopt SkillsBench-style reporting: include **task/domain-level variance** and explicitly count **regressions** (negative deltas), not just averages.
- [ ] If applicable, consider whether “focused” interventions (few modules) outperform “kitchen sink” instructions.

## Quotes / details to potentially cite

- “Curated Skills raise average pass rate by 16.2 percentage points(pp) ... effects vary widely by domain ... and 16 of 84 tasks show negative deltas.”
- “Self-generated Skills provide no benefit on average ... models cannot reliably author the procedural knowledge they benefit from consuming.”
- “Focused Skills with 2–3 modules outperform comprehensive documentation, and smaller models with Skills can match larger models without them.”
