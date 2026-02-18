# Drift No More? Context Equilibria in Multi-Turn LLM Interactions

- Year: 2025
- Venue: arXiv (HTML v1)
- Authors: Vardhan Dongre, Ryan A. Rossi, Viet Dac Lai, David Seunghyun Yoon, Dilek Hakkani-Tür, Trung Bui
- URL: https://arxiv.org/html/2510.07777v1
- BibTeX key (if we add it): dongre2025drift
- Tags: multi-turn, context-drift, dynamical-systems, equilibrium, reminders, tau-bench, user-simulators, KL-divergence

## One-sentence takeaway

Multi-turn “context drift” (measured as KL divergence to a goal-consistent reference model) tends to stabilize at a noise-limited equilibrium rather than growing unboundedly, and simple goal-reminder interventions shift that equilibrium downward.

## What problem does it solve?

- How to *measure* and *reason about* context drift over many dialogue turns (beyond per-turn accuracy/end-task success).
- How to interpret whether drift is “inevitable decay” vs a bounded process, and how light interventions affect long-run behavior.

## What is the core method / protocol?

- Define turn-wise contextual divergence:
  - At turn t, compare test model distribution q_t(y) = P_\theta(y | x_<t) vs reference distribution p_t(y) = P*(y | x_<t).
  - Drift proxy: D_t = D_KL(q_t || p_t).
- Propose a simple recurrence / stochastic-process view:
  - D_{t+1} = D_t + g_t(D_t) + \eta_t - \delta_t,
  - where g_t captures systematic bias/memory effects, \eta_t bounded noise, and \delta_t >= 0 an “intervention” (e.g., reminders).
  - Interpret long-run behavior via a stable equilibrium D* (fixed point in expectation).
- Instantiate on two settings:
  - Synthetic long-horizon constrained rewriting with increasingly conflicting instructions.
  - Realistic goal-driven multi-turn user–agent simulations using τ\tau-bench; they study open-weight LLMs as *user simulators* and use GPT-4.1 as the goal-consistent reference.
- Evaluate reminders: inject goal restatements at selected turns (e.g., t=4 and t=7).

## What are the key metrics?

- Primary: per-turn KL divergence D_t to the reference policy; also mentions JS divergence.
- Secondary triangulation: semantic similarity (Sim) and an LLM-judge alignment score (Likert 1–5) conditioned on goal/profile/history.
- Diagnostic regression to estimate equilibrium: fit ΔD_t = a + b D_t + noise; equilibrium estimate D* = -a/b (with b < 0 as “restoring force”).

## What are the main results?

- Empirically, drift trajectories are typically bounded and fluctuate around a stable, model-specific equilibrium (not monotone blow-up with turn count).
- Reminders reliably reduce divergence and improve judged alignment, consistent with the “equilibrium shift” view.
- Interprets the effect as controllable dynamics: interventions lower the equilibrium divergence level rather than eliminating drift.

## How is this similar to GALILEO?

- Same core theme: *multi-turn robustness / stability over time* and the need for metrics that respect temporal dynamics.
- Provides an explicitly dynamical framing (equilibria, restoring forces, interventions) that can complement GALILEO’s survival/flip/recovery style reporting.

## How is this different from GALILEO?

- Drift is defined relative to a *reference model distribution* (GPT-4.1), not primarily via outcome-based correctness/flip events under social pressure.
- Focuses on user-simulator drift in τ\tau-bench and synthetic constraint-following tasks; less directly about pressure/persuasion operators and belief-revision vs drift controls.
- Requires access to (or approximation of) token-level predictive distributions for KL, which may be harder to compute consistently across closed models.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO centers pressure-vs-evidence controls and recovery-after-flip trajectories, it can make a cleaner causal distinction than “divergence to a strong anchor.”
- GALILEO-style metrics (time-to-failure, recovery rate, oscillation patterns) are more directly interpretable than KL-to-reference for many readers.

## Where GALILEO is weaker / needs to improve

- GALILEO may benefit from an explicit *equilibrium/stationarity* story for long-horizon behavior (not just “it fails by turn k”).
- If GALILEO lacks a continuous-valued “drift coordinate,” this paper is a strong citation for introducing one (even if we choose a different definition than KL-to-GPT).

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a short “bounded drift / equilibrium” framing paragraph in related work: drift need not grow unbounded; interventions can shift a steady-state level.
- [ ] Consider adding an equilibrium-estimation diagnostic for our own turn-wise metric(s): regress Δm_t on m_t to estimate a fixed point and show reminder/intervention shifts.
- [ ] If we already use reminders or goal restatements, cite this as evidence that lightweight reminder controls can reliably improve long-horizon alignment.

## Quotes / details to potentially cite

- Abstract-level claim: drift “reveals stable, noise-limited equilibria rather than runaway degradation,” and “simple reminder interventions reliably reduce divergence.”
- Definition: “We formalize drift as the turn-wise KL divergence between the token-level predictive distributions of the test model and a goal-consistent reference model.”
- Equilibrium interpretation: treat D* as the long-run divergence sustained under a protocol; interventions shift D* downward.
