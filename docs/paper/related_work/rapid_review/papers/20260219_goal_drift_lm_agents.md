# Technical Report: Evaluating Goal Drift in Language Model Agents

- Year: 2025
- Venue: arXiv (technical report)
- Authors: Rauno Arike; Elizabeth Donoway; Henning Bartsch; Marius Hobbhahn
- URL: https://arxiv.org/abs/2505.02709v1 (HTML: https://arxiv.org/html/2505.02709v1)
- BibTeX key (if we add it): arike2025goald drift (suggested)
- Tags: agents, goal-drift, long-context, robustness, evaluation, safety

## One-sentence takeaway

A controlled stock-trading benchmark + metrics for measuring *inference-time* goal drift in LM agents under long contexts, adversarial nudges, and goal-switching, showing drift exists for all tested models and grows with long-context pattern-matching.

## What problem does it solve?

- How to *detect and quantify* gradual deviation from an instruction-specified objective (“system goal”) as an agent runs for long horizons without human oversight.
- How to operationalize “goal drift” behaviorally (not via self-report) and separate it from baseline preferences.

## What is the core method / protocol?

- Simulated **stock trading environment** (portfolio manager) with tool-like actions (buy/sell/check stats) executed over many “quarters” (time steps).
- Assign a **system goal** in the system prompt (e.g., minimize emissions) and define a **competing goal** (e.g., maximize profit) with mutually exclusive stock clusters.
- Evaluate across settings:
  - **Baseline**: no adversarial pressures.
  - **Adversarial pressure**: environmental nudges (news/emails/distractor questions/positive reinforcement once drift begins) that encourage the competing goal without directly instructing a goal change.
  - **Goal switching**: instruct agent to first pursue an *instrumental* goal, then switch back to the system goal at a detectable completion condition (early intrinsification indicator).
  - **Combined**: adversarial pressure during the post-switch evaluation phase.
- Define goal drift scores as deviations from baseline, measured via **actions** (commissions: buying misaligned stocks / saving cash when it should invest) and **inaction** (omissions: failing to sell instrumental-goal-aligned stocks after switch).
- Study very long contexts (claimed >100k tokens) and run ablations to probe mechanisms (pattern-matching susceptibility increasing with depth in context).

## What are the key metrics?

- Goal drift through actions: relative reduction in fraction of budget invested in system-goal-aligned cluster vs baseline.
- Goal drift through inaction (in goal-switching settings): failure to divest from instrumental-goal-aligned cluster after system-goal phase begins.
- (Implicit) performance/portfolio outcomes in the simulator as downstream behavior signal.

## What are the main results?

- All evaluated agents show **some goal drift** under at least some settings.
- A scaffolded **Claude 3.5 Sonnet** variant shows near-perfect adherence for **>100k tokens** even in the hardest setting, but is not completely immune.
- Drift correlates with **increasing pattern-matching** behavior as context length grows.

## How is this similar to GALILEO?

- Targets *long-horizon agent robustness* to staying on-task under multi-turn interaction pressure.
- Frames misbehavior as a **behavioral deviation** from an intended objective in an interactive environment.
- Provides language and citations around “goal drift” / long-context degradation that can motivate GALILEO-style evaluations.

## How is this different from GALILEO?

- Uses a **domain-specific simulated trading environment** with numeric portfolio actions and hand-constructed adversarial “pressure” texts, rather than open-ended real tasks.
- Operationalizes drift via **cluster-aligned investments** (a fairly narrow action space); may not capture richer task-level drift (scope creep, tool misuse, spec violations) depending on GALILEO’s target behaviors.
- Includes an explicit **goal-switching** (instrumental → system) setup aimed at early intrinsification indicators.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO evaluates goal adherence on *realistic tasks / tool use / constraints*, it may offer broader external validity beyond a single simulated domain.
- If GALILEO provides more fine-grained taxonomies of drift (scope creep, policy violations, spec divergence), it may be more actionable for developers.

## Where GALILEO is weaker / needs to improve

- This paper provides a fairly crisp **experimental manipulation** (pressure vs baseline; switching) and long-context token scaling; GALILEO should ensure similarly clean causal comparisons and long-context sweeps.
- Consider adding a **goal-switching / post-instrumental divestment** style test if GALILEO doesn’t already include it.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider citing this as a definition/measurement reference for “goal drift” (behavioral, inference-time) and as motivation for long-horizon evaluations.
- [ ] If applicable, add a GALILEO ablation: drift vs **context depth** / trajectory length, and test whether drift correlates with **pattern-matching** signals.
- [ ] Consider a GALILEO setting analogous to **goal switching** (temporary subgoal then return) to probe early intrinsification-like failures.

## Quotes / details to potentially cite

- Abstract (paraphrase-worthy): goal drift is subtle/gradual; evaluate agents given a system goal then exposed to competing objectives via environmental pressures; best model maintains adherence for >100k tokens but all models show some drift; drift correlates with pattern-matching as context grows.
