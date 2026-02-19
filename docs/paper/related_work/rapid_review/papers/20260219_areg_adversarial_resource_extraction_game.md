# AREG: Adversarial Resource Extraction Game for Evaluating Persuasion and Resistance in Large Language Models

- Year: 2026
- Venue: arXiv
- Authors: Adib Sakhawat, Fardeen Sadab
- URL: https://arxiv.org/abs/2602.16639
- BibTeX key (if we add it): areg2026sakhawat
- Tags: persuasion, resistance, multi-turn, benchmark, negotiation, LLM-as-judge, Elo

## One-sentence takeaway

AREG is a multi-turn, zero-sum “resource extraction” negotiation game that separately scores LLMs’ persuasion (offense) and resistance (defense) via an automated arbiter and dual Elo ratings, finding these capabilities are only weakly correlated and that defense tends to dominate.

## What problem does it solve?

- Existing persuasion evals often measure *static* text quality rather than *interactive* outcomes in adversarial dialogue.
- “Resistance” is often measured as stance/belief change (subjective) rather than concrete, verifiable outcomes.
- It is unclear whether persuasion and resistance are the same latent capability (they test if they dissociate).

## What is the core method / protocol?

- Define a finite-horizon (Tmax = 10) asymmetric, incomplete-information dialogue game:
  - Culprit (persuader) tries to extract money.
  - Victim (defender) starts with a private budget B0 = $100 and tries to retain it.
  - Culprit cannot see the Victim’s budget; Victim can.
- A deterministic Arbiter (LLM at temperature 0) parses the Victim’s messages each turn to identify *new unconditional* monetary commitments and updates the budget:
  - Only immediate, explicit “here is $X” style transfers count.
  - Future promises / conditional offers / ambiguous assent do not count.
  - Incremental deltas are recognized to avoid double-counting.
- Run a round-robin tournament among 8 frontier models (multiple providers) for 5 rounds:
  - Each pair plays in both roles; total 280 games.
- Score with a continuous-outcome “Linear Elo” using extraction ratio S = (100 - B_final)/100, maintaining two ratings per model:
  - C-Elo for persuasion (Culprit role)
  - V-Elo for resistance (Victim role)

## What are the key metrics?

- Extraction ratio S in [0, 1] derived from final remaining budget.
- Dual Elo ratings (C-Elo and V-Elo) updated using continuous S rather than win/loss.
- Correlation between models’ C-Elo and V-Elo (reported ~0.33).
- Linguistic/strategy analysis signals (e.g., incremental commitment-seeking vs verification-seeking).

## What are the main results?

- Persuasion and resistance are weakly correlated (rho ~ 0.33): good persuaders are not reliably good defenders (and vice versa).
- Across evaluated models, resistance scores exceed persuasion scores: systematic defensive advantage in this adversarial dialogue setup.
- Qualitative/linguistic findings:
  - Incremental commitment-seeking strategies correlate with higher extraction success.
  - Verification-seeking responses are more prevalent in successful defenses than explicit refusal.

## How is this similar to GALILEO?

- Focuses on *interactive*, multi-turn evaluation where outcomes depend on dialogue dynamics rather than single-shot generations.
- Emphasizes adversarial settings and role-conditional behavior (attack vs defend), which aligns with evaluating agentic/social failure modes.
- Uses structured scoring from unstructured interaction traces (LLM-as-judge style adjudication).

## How is this different from GALILEO?

- AREG’s task is narrowly framed as monetary negotiation/resource extraction with a single private state variable (budget), whereas GALILEO (as a broader eval framework) may target wider capability/safety axes and more realistic agent environments.
- Evaluation relies on an Arbiter that detects explicit money-transfer commitments; this is a particular operationalization that may not transfer to other “persuasion” contexts (e.g., policy compliance, tool misuse).
- Uses dual Elo as the headline metric; GALILEO may prefer more directly interpretable success/failure rates, harm scores, or calibrated risk metrics.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO evaluates broader agentic tasks or more realistic interaction constraints, it can better capture generalization beyond a single game mechanic.
- If GALILEO uses stronger ground-truth checks (e.g., environment-state verification or tool logs), it may avoid dependence on judge parsing accuracy.

## Where GALILEO is weaker / needs to improve

- Consider explicitly separating “offense” and “defense” capabilities (AREG’s dual-rating idea) if GALILEO currently reports a single scalar score that can hide asymmetries.
- Consider incorporating analysis of *interaction structure* (turn-by-turn commitment building, verification behavior) rather than only end outcomes.

## Action items for GALILEO (experiments / method / writing)

- [ ] In related work, cite AREG as evidence that persuasion and resistance can dissociate (rho ~ 0.33) and that defensive advantage is common in adversarial dialogue.
- [ ] Consider adding a role-asymmetric reporting section (attack vs defend) or a dual-metric summary for social/adversarial interaction tasks.
- [ ] Add a brief discussion: “verification-seeking” as an effective defense strategy vs blunt refusal, and how GALILEO tasks/metrics might capture that.

## Quotes / details to potentially cite

- “Evaluating the social intelligence of Large Language Models (LLMs) increasingly requires moving beyond static text generation toward dynamic, adversarial interaction.”
- AREG operationalizes persuasion/resistance as “a multi-turn, zero-sum negotiation over financial resources.”
- Capabilities are “weakly correlated (rho = 0.33)” and “resistance scores exceed persuasion scores,” suggesting a defensive advantage.
- Arbiter policy: only *new unconditional* immediate commitments count; conditional/ambiguous language is scored as 0 to prioritize precision.