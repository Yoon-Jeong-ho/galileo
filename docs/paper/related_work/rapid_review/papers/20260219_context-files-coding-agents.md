# Evaluating AGENTS.md: Are Repository-Level Context Files Helpful for Coding Agents?

- Year: 2026
- Venue: arXiv (cs.SE, cs.AI) (ICML listed in arXiv HTML metadata)
- Authors: Thibaud Gloaguen (et al.)
- URL: https://arxiv.org/abs/2602.11988
- BibTeX key (if we add it): Gloaguen2026AgentsContextFiles
- Tags: agents, software-engineering, benchmarks, repo-context, evaluation, instructions

## One-sentence takeaway

Repository-level “agent context files” (e.g., AGENTS.md / CLAUDE.md) often *hurt* coding-agent task success while increasing cost, and the paper argues they should be minimal and avoid unnecessary requirements.

## What problem does it solve?

- There is widespread industry advice to add repo-specific context files for coding agents, but little rigorous evidence that they help on real tasks.
- Two evaluation gaps the paper targets:
  - Existing benchmarks (e.g., SWE-bench) predate widespread context-file adoption, so they lack developer-provided context files.
  - Popular benchmark repos may not represent typical/niche repos where developer-committed context files are used.

## What is the core method / protocol?

- Evaluate coding agents on issue/PR-derived tasks under **three settings**:
  1) **Developer-provided** context file (when present in the repo).
  2) **No context file**.
  3) **LLM-generated** context file created following agent-developer recommendations.
- Two complementary evaluation sources:
  - SWE-bench Lite tasks (popular repos) with **LLM-generated** context files.
  - A new benchmark, **AGENTbench**, mined from repos that already contain developer-committed context files.
- AGENTbench construction (high level):
  - Mine GitHub repos containing AGENTS.md/CLAUDE.md, focus on Python + runnable tests, and mine PRs.
  - Filter PRs/issues for testable, deterministic changes; build executable environments; standardize task descriptions; generate/validate unit tests when needed.

## What are the key metrics?

- Primary: **task success rate** (patch passes test suite / validation).
- Secondary/behavioral: changes in agent behavior from traces (exploration, testing, file traversal) and **inference cost**.

## What are the main results?

- **Context files tend to reduce success rates** relative to no context, while increasing inference cost by **>20%**.
- Developer-provided context files show at most **marginal improvement** over omitting context (paper reports ~+4% average), while **LLM-generated** context files can slightly degrade performance (paper reports ~-3% average).
- Qualitative/trace-level finding: context files push agents toward broader exploration and more tooling usage, and agents tend to follow the instructions.
- Proposed explanation: context files frequently add **unnecessary requirements**, increasing task difficulty.

## How is this similar to GALILEO?

- Directly relevant if GALILEO’s agent setup uses repository-level instruction/context artifacts (AGENTS.md/CLAUDE.md-equivalents) or studies instruction-following under additional constraints.
- Reinforces a core theme: additional “helpful” context/instructions can induce behavioral shifts (more exploration/cost) and can reduce objective performance.

## How is this different from GALILEO?

- This is primarily **software engineering / coding-agent evaluation** (benchmarks, success rates, cost), not a targeted study of social pressure / sycophancy / drift dynamics.
- Intervention space is mainly “include/omit/generate” context files, not black-box behavioral protocols for resisting pressure while preserving corrigibility.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO distinguishes *helpful evidence-driven updating* vs *harmful pressure-driven drift*, that kind of causal disentanglement is typically clearer than a broad “success rate” delta attributable to context-file presence.

## Where GALILEO is weaker / needs to improve

- If GALILEO recommends rich repository-level instruction scaffolding (or evaluates under heavy instruction load), this paper is a warning that **extra requirements can backfire**.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a discussion/related-work note: “Context files/instructions can reduce task success while increasing cost; keep requirements minimal.”
- [ ] If GALILEO uses any persistent instruction headers, run an ablation: **minimal vs verbose** instruction blocks; measure cost + success + failure modes.
- [ ] Consider classifying “instruction overhead” as a confounder when comparing protocols (long prompts can change exploration/cost).

## Quotes / details to potentially cite

- Paper claim (abstract): context files “tend to reduce task success rates compared to providing no repository context, while also increasing inference cost by over 20%.”
- Paper recommendation (abstract): “unnecessary requirements from context files make tasks harder” and context files should be **minimal**.
