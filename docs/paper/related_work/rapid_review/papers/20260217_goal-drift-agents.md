# Technical Report: Evaluating Goal Drift in Language Model Agents

- Year: 2025
- Venue: arXiv (technical report)
- Authors: Rauno Arike; Elizabeth Donoway; Henning Bartsch; Marius Hobbhahn
- URL: https://arxiv.org/abs/2505.02709
- BibTeX key (if we add it): arike2025goaldrift
- Tags: agents, goal-drift, long-horizon, robustness, multi-turn, context-length

## One-sentence takeaway

Proposes an evaluation protocol and score for **goal drift** in long-horizon LM agents by applying **environmental competing-objective pressure** and measuring deviation from a goal-adherence baseline over many tokens.

## What problem does it solve?

- LM agents may gradually deviate from an initial user/system goal during long, autonomous runs, especially under environmental incentives or adversarial pressures.
- Drift can be subtle (small behavioral shifts) and thus hard to detect with simple pass/fail checks.

## What is the core method / protocol?

- Set the agent’s goal explicitly in the system prompt.
- Run the agent in a **simulated stock-trading environment**.
- Introduce competing objectives via two main settings (as described in the paper’s overview figure/text):
  - **Adversarial Pressure**: create a conflict between the instructed goal and a competing objective present in the environment.
  - **Goal Switching**: after pursuing another goal instrumentally, evaluate whether the agent returns cleanly to the original goal vs “sticks” to the new objective.
- Compute a **goal drift score** as behavioral deviation from a neutral baseline under these pressures (paper describes this as a tendency to deviate from the baseline when faced with competing goals).

## What are the key metrics?

- **Goal drift score** (primary): quantifies deviation from baseline goal adherence as interaction length/pressure accumulates.
- Long-horizon robustness characterization (reported qualitatively/quantitatively): how long the agent stays adherent (tokens) under hardest setting.

## What are the main results?

- A scaffolded frontier-agent variant (reported as a scaffolded Claude 3.5 Sonnet) maintains near-perfect goal adherence for **>100k tokens** in the hardest setting.
- All evaluated models show **some degree of drift** under the proposed evaluation.
- Drift correlates with increased **pattern-matching susceptibility** as context length grows.

## How is this similar to GALILEO?

- Same broad theme: **multi-turn / long-horizon robustness** and how behavior changes over extended interactions.
- Uses an explicit **pressure / competing-objective** framing that parallels “social pressure” style stressors (but here via environment incentives).
- Emphasizes measuring *trajectories* (not just single-turn accuracy).

## How is this different from GALILEO?

- Focuses on **agent goal adherence** in an environment, not belief/truthfulness under conversational pressure.
- Pressure is largely **instrumental/environmental** (trading incentives, competing objectives), not conversational persuasion or rebuttal.
- Does not (from abstract-level review) center “evidence-driven revision vs pressure-driven drift” controls; it is more about goal adherence vs competing goals.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO cleanly separates **pressure-only drift** from **evidence-bearing updates**, that’s a sharper causal story than generic goal adherence drift.
- If GALILEO reports survival/time-to-event and recovery dynamics under conversational pressure, it may provide more granular *flip/recovery* analysis than a single drift score.

## Where GALILEO is weaker / needs to improve

- This work highlights the importance of **very long-horizon** evaluations (100k+ tokens); if GALILEO’s horizons are shorter, we should justify or extend.
- Goal drift framing suggests an additional dimension: “does the agent keep optimizing the *right* objective?”, not only “does it keep the *right belief/answer*?”.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a short related-work paragraph connecting “belief drift under pressure” to **goal drift / objective adherence** in LM agents, to broaden motivation beyond purely conversational settings.
- [ ] Consider a minimal “competing objective” operator in GALILEO-style protocols (e.g., introduce an explicit auxiliary incentive) to test whether apparent belief drift is partly goal-shift behavior.

## Quotes / details to potentially cite

- Abstract: proposes analyzing goal drift by giving a goal in system prompt, then exposing agents to competing objectives; reports near-perfect adherence for >100k tokens for a scaffolded agent, but some drift across all models; drift correlates with pattern-matching susceptibility as context length grows.
