# OpenDeception: Learning Deception and Trust in Human–AI Interaction via Multi-Agent Simulation

- Year: 2025
- Venue: arXiv (paper notes mention ICML)
- Authors: Yichen Wu et al.
- URL: https://arxiv.org/abs/2504.13707
- BibTeX key (if we add it): opendeception2025
- Tags: deception, trust, multi-turn, evaluation, simulation, safety

## One-sentence takeaway

OpenDeception proposes a joint, turn-by-turn evaluation of (a) model deceptive intent and (b) user trust/susceptibility in multi-turn dialogues, using LLM-based simulation plus contrastive training to build scalable detectors and early-warning triggers.

## What problem does it solve?

- Existing deception evaluations are often (i) scenario/task-specific (e.g., “did the model cause action X?”) or (ii) model-centric (detect deceptive intent) while ignoring the user side.
- In real deployments, harm depends on *interaction* effects: deceptive behaviors are most dangerous when paired with high user trust / reliance.
- Data scarcity + ethical constraints make it hard to collect real high-risk deceptive dialogues and to label “trust” reliably as an absolute scalar.

## What is the core method / protocol?

- **Benchmark**: a scenario set of **50 real-world deception cases** (grouped into multiple deception types).
- **Two evaluators used jointly during a dialogue**:
  - **IntentNet**: predicts whether the agent is acting with deceptive intent, using the agent’s “reasoning” as input (paper positions this as inferring intent from internal reasoning signals).
  - **TrustNet**: estimates user susceptibility/trust (how much the user believes/relies on the agent).
- **Synthetic data pipeline** (to address scarcity): LLM-driven role-and-goal simulation to generate “high-risk” multi-turn dialogues under controlled settings (roles, incentives, information asymmetry).
- **Trust as contrastive learning** (to avoid noisy scalar labels): construct controlled pairs of user responses that correspond to higher vs lower trust under matched contexts, then train a user-trust scorer with a preference/contrastive objective.
- **Use**: compute deception+trust over turns; trigger warnings when trust crosses unsafe thresholds in presence of deceptive intent.

## What are the key metrics?

- IntentNet accuracy for deceptive-intent detection (reported **>93%**).
- TrustNet accuracy for trust/susceptibility scoring (reported **>77%**).
- “Risk prevalence” style analysis across models: fraction of goal-driven interactions flagged as deceptive intent.

## What are the main results?

- Across **11 LLMs + 3 large reasoning models**, they report **>90%** of goal-driven interactions in most models exhibit deceptive intent.
- They claim *capability correlates with risk*: stronger models show higher deceptive-intent risk.
- A qualitative case study adapted from a documented severe real-world incident is used to illustrate early warning before a “critical trust threshold” is reached.

## How is this similar to GALILEO?

- Multi-turn, trajectory-level evaluation rather than single-turn scoring.
- Emphasizes **dynamics over turns** and detecting “unsafe regimes” early.
- Uses a **two-signal decomposition** (intent-like vs user-state-like) reminiscent of separating “model behavior” from “interaction context / user effects.”

## How is this different from GALILEO?

- Focus is **deception + trust** (human susceptibility) rather than sycophancy / agreement robustness per se.
- Relies on access to an “agent reasoning” channel for IntentNet (may be unavailable or policy-restricted in many settings).
- TrustNet is trained on simulated, contrastive preference pairs; GALILEO may aim for task-agnostic robustness metrics or protocol-based evaluation not dependent on synthetic trust labels.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO avoids dependence on chain-of-thought / “reasoning traces,” it may be easier to apply across closed models and safer to deploy.
- GALILEO-style protocols can be framed without requiring user-susceptibility modeling, reducing confounds and ethical concerns.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently ignores user-state (trust/reliance), it may miss the highest-risk interactions where users are most persuadable or dependent.
- Might need an explicit “risk multiplier” term capturing user susceptibility (even if approximated via behavioral proxies rather than a learned TrustNet).

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding a *joint-risk view* section: risk = (undesired model behavior) × (user susceptibility proxy), even if susceptibility is approximated (e.g., via user compliance markers).
- [ ] In related work: cite OpenDeception as an example of **bidirectional** evaluation (agent intent + user trust) and **contrastive** modeling of unobservable user states.
- [ ] Stress-test whether “stronger model → higher risk” also appears for GALILEO’s target behaviors (sycophancy/pressure collapse).

## Quotes / details to potentially cite

- “OpenDeception … jointly evaluating deception risk from both sides of human-AI dialogue … a scenario benchmark with 50 real-world deception cases, an IntentNet … and a TrustNet … user susceptibility.”
- “Experiments … show that over 90% of goal-driven interactions in most models exhibit deceptive intent … stronger models displaying higher risk.”
- Trust labeling framing: “train the User Trust Scorer using contrastive learning on controlled response pairs, avoiding unreliable scalar labels.”
