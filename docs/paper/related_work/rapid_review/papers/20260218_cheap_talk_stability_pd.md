# Communication Enhances LLMs' Stability in Strategic Thinking

- Year: 2026
- Venue: arXiv
- Authors: Nunzio Lorè; Babak Heydari
- URL: https://arxiv.org/abs/2602.06081
- BibTeX key (if we add it): lore2026communication
- Tags: stability, multi-turn, strategic-thinking, cheap-talk, repeated-games, prisoners-dilemma

## One-sentence takeaway

Allowing agents to exchange short, non-binding “cheap-talk” messages before acting in a repeated Prisoner’s Dilemma reduces *trajectory noise* (i.e., makes cooperation/defection dynamics more stable) across many LLM+context pairings.

## What problem does it solve?

- Multi-agent LLM behavior in strategic settings can be highly *context-dependent* and *variable*, making outcomes hard to predict/reproduce.
- The paper targets **stability/predictability** rather than payoff maximization: “before agents can be aligned to be good, they must first be reliable” (paraphrased).

## What is the core method / protocol?

- Task: **10-round repeated Prisoner’s Dilemma** with two LLM agents.
- Intervention: **pre-play messages** (costless, non-binding; “cheap talk”) vs **no messaging**.
- Scope (as described):
  - Models in the **7B–9B** range.
  - Multiple **prompt variants**, **decoding regimes** (including a temperature-0 robustness check), and multiple **context framings**.
  - Additional extensions to **network interaction** settings (and communication constraints) are mentioned.

## What are the key metrics?

- “Cooperation trajectories” over the 10 rounds.
- Stability quantified by comparing a smoothed trend (LOWESS) to the observed trajectory:
  - Fit LOWESS curves to cooperation vs round.
  - Compute an error/noise measure (described as **RMSE** of smoothed trajectories) and compare between messaging vs no-messaging.
- Uses simulation-level **bootstrap resampling** + **nonparametric inference** to compare conditions.

## What are the main results?

- **Messaging reduces trajectory noise** (improves stability) in a **majority** of model–context pairings.
- Effect is robust across:
  - multiple prompt variants
  - multiple decoding regimes
  - a temperature=0 setting (reported as a robustness check)
- The benefit is **heterogeneous**:
  - larger gains for models with higher baseline volatility
  - depends on model choice and contextual framing
- Communication “rarely” harms stability, but there are **context-specific exceptions** where it can increase instability.

## How is this similar to GALILEO?

- Shared focus on **multi-turn dynamics** and **stability/instability** as a first-class evaluation target.
- Emphasizes that *small prompt/context changes* can yield qualitatively different trajectories—aligned with GALILEO’s concern about drift/flip behavior under interaction.

## How is this different from GALILEO?

- Strategic-game setting (repeated PD) rather than (presumably) belief revision / pressure / persuasion / tool-use drift.
- Primary outcome is **trajectory smoothness/noise** of cooperation rates, not:
  - time-to-failure under pressure
  - separating evidence-driven revision vs pressure-driven drift
  - recovery-after-flip trajectories
- Stabilization lever is **cheap-talk communication** (a protocol change) rather than evaluation controls/metrics designed to diagnose drift vs revision.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO includes explicit **controls** for “new evidence vs social pressure” and explicit **recovery** measurements, it provides a cleaner causal story than “messaging vs not” in a strategic game.
- If GALILEO uses **time-to-event / flip / survival-style** metrics, those can capture *when* instability happens, not only overall trajectory noise.

## Where GALILEO is weaker / needs to improve

- GALILEO may under-explore **communication-channel design** as a low-cost stabilizer (pre-play messaging / coordination talk) that can reduce variance even without changing model weights.
- Need to be careful to distinguish “stability” definitions:
  - this paper’s stability = reduced *trajectory noise* (LOWESS/RMSE)
  - GALILEO may emphasize *truth/consistency* under pressure.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a brief related-work paragraph: cheap-talk style **communication as a stabilizer** for multi-agent LLM trajectories (cite this as evidence that protocol changes can reduce volatility).
- [ ] Consider adding an auxiliary metric analogous to “trajectory noise” (e.g., deviation from a smoothed/expected path) alongside flip/time-to-event metrics.
- [ ] If GALILEO has “communication” variants, include a warning: communication usually stabilizes but can have **context-specific destabilizing** cases.

## Quotes / details to potentially cite

- Abstract-level claim: “short, costless pre-play messages” reduce “trajectory noise” for repeated PD LLM agents.
- Method detail: compare cooperation trajectories with LOWESS regression; use bootstrap resampling + nonparametric inference; report RMSE differences between conditions.
- Framing line (intro-level): stability is a prerequisite for alignment (“before agents can be good, they must be reliable”).
