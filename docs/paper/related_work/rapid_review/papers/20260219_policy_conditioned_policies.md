# Policy-Conditioned Policies for Multi-Agent Task Solving

- Year: 2025
- Venue: arXiv (cs.GT, cs.AI)
- Authors: Shuhui Zhu; Wenhao Li; Ang Li; Dan Qiao; Pascal Poupart; Hongyuan Zha; Baoxiang Wang
- URL: https://arxiv.org/abs/2512.21024
- BibTeX key (if we add it): zhu2025policyconditioned
- Tags: multi-agent, marl, programmatic-policies, llm, best-response, program-equilibrium

## One-sentence takeaway

Represent multi-agent policies as readable source code and use an LLM to iteratively synthesize/debug a point-wise best response (via “textual gradients” + unit tests), approximating a practical form of Program Equilibrium.

## What problem does it solve?

- In MARL, adapting to opponents is hard because (a) opponents’ policies drift (non-stationarity) and (b) “opponent modeling” from interaction history is brittle.
- The paper highlights a “representational bottleneck”: conditioning a policy on another agent’s policy is intractable when policies are opaque neural-network parameter vectors.
- Goal: enable policy-conditioning / best-response computation when agents can observe each other’s policies.

## What is the core method / protocol?

- **Programmatic policies**: represent each agent’s policy as executable, human-readable source code instead of NN weights.
- Use an **LLM as an approximate interpreter / best-response operator** that maps opponent code → ego code.
- Proposed training dynamic: **Programmatic Iterated Best Response (PIBR)**
  - Outer loop alternates which agent updates.
  - Inner loop “optimizes” the current agent’s policy code against fixed opponent code.
  - Feedback signals:
    - **Unit-test / runtime feedback**: ensure generated code is syntactically valid and executable (use stack traces / failures as high-priority correction signals).
    - **Utility feedback**: roll out games/episodes and feed trajectories/rewards back to guide improvements.
  - Uses the framing of **textual gradients** (cites TextGrad-style optimization) to refine prompts/code based on structured feedback.

## What are the key metrics?

- **Social welfare / joint return** (sum of cumulative rewards across agents) on:
  - Coordination matrix games (Vanilla Coordination, Climbing Game, Penalty Game).
  - Cooperative Level-Based Foraging (a modified fully cooperative variant with terminal penalty encouraging efficiency).
- Qualitative metric: frequency of **non-executable policies** (missing datapoints when code errors prevent evaluation).

## What are the main results?

- In standard **coordination matrix games**, PIBR quickly reaches the globally optimal coordination outcomes (often within ~one update).
- In **cooperative foraging**, performance improves but is less stable: high variance, occasional near-optimal episodes, and later-stage drops; they select the best-performing iterate from history.
- Takeaway: the approach works well in small/structured games; robustness remains challenging in more complex environments.

## How is this similar to GALILEO?

- Treats LLMs as components in an **agentic control loop** (generate candidate policy/program, execute/evaluate, feed back structured signals, iterate).
- Emphasizes **verification hooks** (unit tests / runtime checks) as guardrails for LLM-generated artifacts.
- Aligns with the idea that **interpretable artifacts** (code, specs) can be a better interface than opaque latent policies.

## How is this different from GALILEO?

- Focus is specifically on **multi-agent best response / equilibrium seeking** in Markov games, rather than GALILEO’s primary setting (as framed in our paper) of end-to-end system design / evaluation.
- Assumes a setting where agents can access **opponent source code** (or an equivalent “open-source games” style transparency), which may not match GALILEO’s assumptions.
- Optimization is described as **operator/prompt/code optimization** for policies, not (necessarily) learning generalizable components across tasks.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO does not require full opponent-code observability, it is likely **more realistic** for many deployment settings.
- If GALILEO has more explicit guarantees/structure around evaluation protocols, it may be cleaner than selecting “best iterate from history” post hoc.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently lacks a crisp story about **policy-conditioning / opponent transparency**, this paper provides a concrete framing and vocabulary (program equilibrium, best-response operator, representational bottleneck).
- If GALILEO does not include strong **runtime validation** loops, PIBR’s unit-test feedback is a good pattern to cite.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider a short related-work paragraph on **Program Equilibrium / open-source games** + “representational bottleneck” as motivation for interpretable policy representations.
- [ ] If relevant, add a comparison point: **LLM-in-the-loop policy/program optimization with runtime tests** (unit-test feedback) as an instance of structured feedback beyond reward.
- [ ] Decide whether to position GALILEO as (a) requiring code transparency, (b) not requiring it, or (c) supporting both; clarify in writing.

## Quotes / details to potentially cite

- Abstract-level claim: deep RL has a “representational bottleneck” because neural policies are “opaque, high-dimensional parameter vectors” that other agents can’t practically condition on.
- Method summary: PIBR “optimizes policy code by textual gradients, using structured feedback derived from game utility and runtime unit tests.”
