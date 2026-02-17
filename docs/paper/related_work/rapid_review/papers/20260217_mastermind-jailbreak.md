# Knowledge-Driven Multi-Turn Jailbreaking on Large Language Models

- Year: 2026
- Venue: arXiv
- Authors: Songze Li, Ruishi He, Xiaojun Jia, Jun Wang, Zhihui Fu
- URL: https://arxiv.org/abs/2601.05445
- BibTeX key (if we add it): mastermind2026knowledge
- Tags: multi-turn, jailbreak, attacks, planning, fuzzing, safety

## One-sentence takeaway

Mastermind is a knowledge-driven, hierarchical multi-agent framework for *adaptive* multi-turn jailbreaking that maintains long-horizon coherence via planning/execution/control and improves over time by distilling and fuzzing reusable attack strategies.

## What problem does it solve?

- Prior multi-turn jailbreaks often (i) lose long-horizon coherence, (ii) follow brittle pre-defined trajectories that break on refusals/deviations, and (iii) depend heavily on manually designed strategies.
- The paper targets *practical* multi-turn attacks that can adapt to a model’s evolving dialogue state over long interactions.

## What is the core method / protocol?

- **Mastermind**: closed loop of **planning → execution → reflection**.
- **Hierarchical multi-agent architecture** (as described in the intro):
  - **Planner**: sets high-level adversarial trajectory / subgoals over many turns.
  - **Executor**: handles local turn-by-turn interaction tactics.
  - **Controller**: monitors dialogue state; triggers tactical redirection/error-correction while preserving overall goal.
  - **Distiller agent**: analyzes *successful* jailbreak trajectories (on a sandbox model) and extracts **abstract, reusable strategies** into a **knowledge repository**.
- **Strategy-level fuzzing**: searches in a *strategy-combination space* (retrieve/recombine/mutate abstract strategies; described as genetic-based fuzzing) instead of raw token-level mutation.

## What are the key metrics?

- Attack success (reported as higher “attack success rates”).
- Harmfulness rating of generated outputs.
- Resilience against advanced defenses (qualitatively stated; likely measured as success under defenses).

## What are the main results?

- Outperforms existing baselines on multiple frontier models (the abstract explicitly names GPT-5 and Claude 3.7 Sonnet), achieving substantially higher attack success and harmfulness ratings.
- Demonstrates notable resilience against multiple advanced defense mechanisms.

## How is this similar to GALILEO?

- Both care about **multi-turn dynamics** where behavior changes across turns and where *stateful* interactions matter.
- The paper’s framing highlights failures of rigid trajectories and need for **long-horizon coherence + local adaptability**—concepts adjacent to multi-turn robustness/stability evaluation.

## How is this different from GALILEO?

- This is an **attack framework** (goal: elicit harmful outputs), not an evaluation method aimed at measuring benign-task robustness.
- Optimizes an adversary’s strategy with a knowledge repository + fuzzing, rather than focusing on controlled evaluation protocols/metrics for drift, inconsistency, or time-to-failure on tasks.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO emphasizes transparent, task-grounded robustness metrics (e.g., time-to-failure, recovery, consistency), that can be easier to interpret than “harmfulness” scoring in adversarial settings.
- GALILEO can provide a *defender/evaluator* perspective, complementing attacker-centric optimization.

## Where GALILEO is weaker / needs to improve

- If GALILEO’s adversarial testing is mostly fixed-script or limited-adaptivity, this paper suggests adaptive, planner/controller-style attacks can be more realistic and more damaging.
- Need to consider **strategy-space adaptivity** (not just prompt perturbations) when claiming multi-turn robustness.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a “**hierarchical adaptive adversary**” threat model section in related work (Planner/Executor/Controller pattern).
- [ ] When discussing multi-turn safety/robustness, explicitly contrast **token/prompt-level mutation** vs **strategy-level** search.
- [ ] Consider a lightweight ablation in our own eval: fixed trajectory vs adaptive controller that can recover after refusals/derailments (even for non-safety tasks), to test whether our metrics are robust to adaptive policies.

## Quotes / details to potentially cite

- Abstract: proposes a “closed loop of planning, execution, and reflection” and a “hierarchical planning architecture that decouples high-level attack objectives from low-level tactical execution”.
- Intro claims prior multi-turn methods drift into benign subtopics, break when refusals occur, and rely on manual strategy design.
