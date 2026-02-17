# SEAD: Self-Evolving Agent for Multi-Turn Service Dialogue

- Year: 2026
- Venue: arXiv
- Authors: Yuqin Dai, Ning Gao, Wei Zhang, Jie Wang, Zichen Luo, Jinpeng Wang, Yujie Wang, Ruiyuan Wu, Chaozheng Wang
- URL: https://arxiv.org/abs/2602.03548
- BibTeX key (if we add it): sead2026
- Tags: agents, self-play, self-evolution, task-oriented-dialogue, multi-turn, curriculum, user-simulation

## One-sentence takeaway

SEAD is a self-evolving training framework for goal-oriented service-dialogue agents that decouples user simulation into (i) an adversarial curriculum/profile controller and (ii) a role-play user model, aiming for *balanced* (not unfair) interactive training without large annotated dialogue logs.

## What problem does it solve?

- Service (task-oriented) multi-turn dialogues are hard to train well with standard supervised fine-tuning because real human logs are scarce, noisy, and biased (e.g., skewed toward failures).
- Naive interactive/self-play user simulators can become “unfair adversaries” (user side can arbitrarily accept/reject), breaking the causal link between agent quality and success.
- User simulators often lack realistic variation (attention lapses, noise, irrationality) and do not adapt difficulty as the agent improves.

## What is the core method / protocol?

- Proposes **SEAD (Self-Evolving Agent for Service Dialogue)** with *decomposed user modeling*:
  - **Profile Controller / profile generator**: samples initial user states and *controls curriculum* by choosing scenarios; this part is the “adversarial” component.
  - **User Role-play Model**: generates user responses conditioned on the scenario/user state, aiming for realistic role-play.
- Key design goal: avoid an “unfair game” by limiting what the user side can manipulate.
- The paper frames the profile generator’s role as selecting “golden training scenarios” where the agent succeeds about half the time (i.e., informative difficulty).
- Application setting (as described): outbound-call service dialogue with enumerated discrete user-state factors (cooperation / emotion / trust) producing 120 initial state combinations.

## What are the key metrics?

- Task completion rate.
- Dialogue efficiency (reported as improved efficiency; details likely include turns/time/cost, but confirm from PDF if needed).
- Additional “multi-level metrics” including user-state understanding and realism are claimed, but specifics are not fully visible from the abstract/HTML excerpt.

## What are the main results?

- Reports improvement over open-source foundation models and closed-source commercial models.
- Claimed gains: **+17.6% task completion rate** and **+11.1% dialogue efficiency**.
- Claims smaller models trained with SEAD can outperform larger/API baselines while reducing costs.

## How is this similar to GALILEO?

- Common theme: **multi-turn interaction dynamics** and measuring/optimizing agent performance over *trajectories*, not single-turn QA.
- Curriculum/controlled environment design resonates with GALILEO’s emphasis on *protocol clarity* and avoiding confounds (here: “unfair adversary” confound in self-play evaluation/training).

## How is this different from GALILEO?

- SEAD focuses on **training** task-oriented service agents via interactive self-evolution; GALILEO is positioned around **robustness/instability measurement** (e.g., drift/sycophancy/turn-of-failure style phenomena) rather than building user simulators for downstream service tasks.
- SEAD’s primary outcomes are task success and efficiency in a specific service setting, not robustness metrics like survival/time-to-failure under pressure or adversarial multi-turn attacks.

## Where GALILEO is stronger / cleaner (if true)

- GALILEO-style contributions can be cleaner on **evaluation protocol + metrics** for instability/robustness across model families, whereas SEAD’s results may be more setting-specific (service SOPs, discrete user-state design, simulator realism assumptions).

## Where GALILEO is weaker / needs to improve

- If we want to discuss *interventions* (not just measurement), SEAD is a useful exemplar of how to engineer the **user/environment side** to avoid degenerate/adversarial dynamics in multi-turn training.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a short related-work paragraph on **self-evolving / self-play training for multi-turn dialogue agents**, emphasizing SEAD’s “avoid unfair adversary” design as an analogy for avoiding confounded multi-turn robustness evaluations.
- [ ] Consider whether our benchmarks need an explicit *environment fairness* criterion: does the “user”/pressure process preserve a causal link between model behavior and success/failure?

## Quotes / details to potentially cite

- “SEAD decouples user modeling into two components: a Profile Controller … and a User Role-play Model … [so] the environment provides adaptive training scenarios rather than acting as an unfair adversary.” (abstract)
- “Experiments … improving task completion rate by 17.6% and dialogue efficiency by 11.1%.” (abstract)
