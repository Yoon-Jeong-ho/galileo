# Drift No More? Context Equilibria in Multi-Turn LLM Interactions

- Slug: drift-no-more
- Year: 2025
- Venue: Personalization in the Era of Large Foundation Models Workshop (arXiv)
- Authors: Vardhan Dongre; Ryan A. Rossi; Viet Dac Lai; David Seunghyun Yoon; Dilek Hakkani-Tür; Trung Bui
- Links:
  - paper: https://arxiv.org/abs/2510.07777
  - html: https://arxiv.org/html/2510.07777v2
  - code (if any): (not found in skim)
- Bibtex: https://doi.org/10.48550/arXiv.2510.07777

## 1) What problem does it study?
“Context drift” in long, multi-turn interactions: the model’s outputs gradually deviate from goal-consistent behavior across turns, in ways that static end-task metrics miss.

Key framing move: measure drift as *turn-wise divergence* between the test model and a **goal-consistent reference policy**.

## 2) Experimental setup (what is being measured?)
- Task(s):
  - Synthetic long-horizon rewriting tasks (controlled setting).
  - Realistic user–agent simulations (mentions c4-Bench) where open-weight LLMs are used as user simulators.
- Perturbation/pressure type:
  - Not “social pressure” / persuasion; rather, long-horizon accumulation of context + noise.
  - Intervention condition: **goal reminders** (lightweight reminder prompts) injected during the interaction.
- Multi-turn? Y (long-horizon; exact turn counts not captured in skim).
- Metrics:
  - Primary: per-turn **contextual divergence**
    - \(D_t = D_{KL}(q_t \| p_t)\) where \(q_t\) is the test LM next-token distribution and \(p_t\) is the reference policy next-token distribution at the same history.
  - Trajectory-level: whether \(D_t\) exhibits stable equilibrium vs unbounded growth; effect of interventions on the equilibrium level.

## 3) Key findings (bullet)
- Empirically, drift trajectories often reach **stable, noise-limited equilibria** rather than monotonically worsening “runaway” degradation.
- A simple stochastic recurrence / dynamical-process view with “restoring forces” can explain the observed stabilization.
- **Reminder interventions** reliably reduce divergence (shift equilibrium downward), consistent with the proposed framework.

## 4) Limitations / threats
- Requires (or assumes access to) a **goal-consistent reference policy** and its token-level distributions; may be expensive or unavailable in many real deployments.
- KL on token distributions is a *proxy* for goal-consistency; it may not correspond cleanly to human-meaningful “goal drift” in all tasks.
- Not targeted at sycophancy/persuasion; no explicit “pressure” operators or belief-revision-vs-drift controls.

## 5) How it relates to GALILEO
- What we can cite it for:
  - A principled, trajectory-native notion of **drift as a dynamical process**.
  - The idea that drift can be **controlled** via interventions (reminders), and that it may stabilize to an equilibrium.
- Where we differ (our delta):
  - GALILEO focuses on *social pressure / persuasion-induced* belief drift/flip and distinguishes that from legitimate evidence-driven revision (and measures recovery), rather than KL divergence to a reference policy.
- Direct mapping:
  - Survival  drift equilibrium curves over turn index (not survival/time-to-event, but similar “trajectory-first” reporting).
  - TOF  “time/turn until divergence exceeds a threshold” (a possible derived metric from \(D_t\)).
  - Recovery  “post-intervention return-to-equilibrium” / relaxation dynamics after reminders.
  - Neutral Re-asking Control  reference-policy comparison at matched histories (conceptually analogous, but operationally different).

## 6) Quote-able lines
- “Our experiments consistently reveal stable, noise-limited equilibria rather than runaway degradation…” (abstract)
- “…simple reminder interventions reliably reduce divergence…” (abstract)

## 7) Actions
- [ ] Add to paper: Related Work  Drift controls / context drift measurement (cite as KL-to-reference + equilibrium framing; 1 sentence).
- [ ] Add to bib
