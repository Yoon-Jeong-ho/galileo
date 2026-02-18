# Pushing Forward Pareto Frontiers of Proactive Agents with Behavioral Agentic Optimization

- Year: 2026
- Venue: arXiv (submitted as "Machine Learning, ICML" per arXiv HTML)
- Authors: Zhepeng Cen; Haohong Lin; Shiqi Liu; Zuxin Liu; Jiacheng Zhu; Zhang-Wei Hong; Laixi Shi; Ding Zhao
- URL: https://arxiv.org/abs/2602.11351
- BibTeX key (if we add it): Cen2026BAOProactiveAgents
- Tags: proactive-agents, agentic-rl, multi-objective-optimization, user-engagement

## One-sentence takeaway

BAO frames proactive agent training as a multi-objective RL problem (task success vs user-engagement effort) and improves the Pareto trade-off by explicitly enhancing + regularizing multi-turn behaviors like retrospective reasoning and prospective planning.

## What problem does it solve?

- In multi-turn proactive agents, optimizing for task reward alone can encourage excessive user interventions/verification requests, hurting user satisfaction.
- Simply reweighting a penalty on user-engagement actions often fails to produce better trade-offs (i.e., does not move the Pareto frontier).

## What is the core method / protocol?

- Formulate proactive agent learning as a contextual MDP with a hidden user context c (preferences/characteristics).
- Split actions into:
  - user-involved actions Au (require explicit user feedback, e.g., verification)
  - environment/tool actions Ae (no user intervention)
- Optimize a multi-objective objective: maximize expected return while minimizing user-involved action count U(tau) via a weight w.
- BAO: Behavioral Agentic Optimization
  - **Behavior enhancement**: enrich multi-turn reasoning/planning behaviors to improve information gathering and use:
    - Retrospective reasoning: (i) memory management; (ii) hypothesis refinement
    - Prospective planning: (i) dynamic scheduling under a finite interaction budget; (ii) strategical querying (reduce uncertainty, avoid repeated/vague asks)
  - **Behavior regularization**: suppress inefficient/redundant interactions and align behavior with user expectations (details not fully captured in the fetched excerpt).

## What are the key metrics?

- Two-objective evaluation on UserRL benchmark tasks:
  - Task performance (e.g., success/pass rate; the paper references Pass@U-k: pass rate allowing up to k user-involved actions)
  - User engagement effort: number of user-involved actions per trajectory (U(tau))
- Pareto frontier / trade-off curves across different weights w.

## What are the main results?

- On multiple UserRL tasks, BAO improves the trade-off between task performance and user engagement versus proactive agentic RL baselines.
- Claims comparable or better performance than commercial LLM agents while requiring similar or less user engagement.

## How is this similar to GALILEO?

- Both care about *multi-turn agent behavior* beyond single-turn accuracy, emphasizing planning/querying and efficient information gathering.
- Both naturally involve a trade-off between outcome quality and interaction cost/burden.

## How is this different from GALILEO?

- BAO is explicitly framed as **multi-objective agentic RL** with a user-engagement action count objective; GALILEO (as positioned in our paper) may emphasize different supervisory signals and/or evaluation regimes.
- BAO operationalizes user burden as "user-involved actions"; GALILEO may measure interaction cost differently (tokens, tool calls, latency, human preference, etc.).

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO provides clearer instrumentation of *which* interactions are valuable (vs redundant) and more directly ties interaction policies to interpretability/controls, we can argue stronger practicality than generic user-action penalties.

## Where GALILEO is weaker / needs to improve

- We should ensure we have a crisp multi-objective story (or an equivalent) and report explicit trade-off curves; BAO’s framing is straightforward to communicate.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add/verify a Pareto-frontier style plot in our paper: task success vs interaction burden (human-involved actions, confirmations, or a proxy).
- [ ] In related work, cite BAO as: multi-objective proactive agent RL; behavior enhancement (retrospective reasoning + prospective planning) + behavior regularization.
- [ ] Consider adopting the "Pass@U-k" style reporting: performance under a cap of k user-involved actions.

## Quotes / details to potentially cite

- "We model this challenge as a multi-objective optimization (MOO) problem, with user engagement and task performance as potentially conflicting objectives."
- Objective definition: maximize E[R(tau) - w U(tau)] where U(tau) counts user-involved actions.
- Multi-turn behaviors named: memory management, hypothesis refinement, dynamic scheduling, strategical querying.
