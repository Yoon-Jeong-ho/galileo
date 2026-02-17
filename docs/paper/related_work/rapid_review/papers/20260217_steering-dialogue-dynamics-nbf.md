# Steering Dialogue Dynamics for Robustness against Multi-turn Jailbreaking Attacks

- Year: 2025
- Venue: TMLR (arXiv)
- Authors: Hanjiang Hu et al.
- URL: https://arxiv.org/abs/2503.00187
- BibTeX key (if we add it): hu2025steering
- Tags: multi-turn, jailbreak, defense, steering, robustness, control-theory, context-drift

## One-sentence takeaway

Frames multi-turn jailbreak defense as an *invariant safety* control problem and uses a learned **neural barrier function** to proactively filter unsafe turns that emerge due to contextual drift.

## What problem does it solve?

- Existing jailbreak defenses often work for *single-turn* harmful prompts but degrade in **multi-turn** settings where the conversation context drifts and eventually enables unsafe outputs.
- Need a defense that provides per-turn guarantees/behavior closer to “never leave the safe set” despite an adaptive adversary across turns.

## What is the core method / protocol?

- Model the dialogue as a **state-space** dynamical system (conversation state evolves turn-by-turn).
- Learn a **Neural Barrier Function (NBF)** that acts like a safety certificate: it predicts whether the current (state, query) is within a safe region and triggers filtering/steering when approaching the boundary.
- “Safety steering” is applied at each turn to maintain **invariant safety** under multi-turn adversarial prompting (mitigating context drift).

## What are the key metrics?

- Safety vs helpfulness trade-off (incl. **over-refusal**).
- Robustness under **multi-turn jailbreaking attacks** (attack success rate / unsafe response rate).
- Comparative evaluation against baselines such as safety alignment, prompt-based steering, and lightweight LLM guardrails.

## What are the main results?

- Across multiple LLMs, NBF-based steering reportedly outperforms:
  - safety alignment alone,
  - prompt steering,
  - lightweight guardrail approaches,
  while maintaining a better safety–helpfulness–over-refusal trade-off.

## How is this similar to GALILEO?

- Shares the core framing that **multi-turn interactions** induce *stateful drift* that can cause failures later in the trajectory.
- Emphasizes **trajectory-level robustness**, not just single-turn correctness/safety.

## How is this different from GALILEO?

- This is primarily a **defense/steering** paper for *jailbreaking* (harmful content elicitation), whereas GALILEO is focused on *measurement/characterization* of multi-turn robustness phenomena (e.g., drift, inconsistency, pressure).
- Uses control-theoretic invariance + barrier functions, rather than evaluation protocols/metrics as the main contribution.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO provides clearer decompositions of drift vs evidence-based revision (or turn-of-failure / survival-style metrics), it may offer more diagnostic granularity than a defense-first framing.

## Where GALILEO is weaker / needs to improve

- Could benefit from importing the **“safe set / invariance”** language as a unifying conceptual lens for multi-turn robustness (beyond jailbreaks), and from evaluating whether interventions keep trajectories within desired regions.

## Action items for GALILEO (experiments / method / writing)

- [ ] Writing: add a short related-work paragraph on **control-theoretic steering / barrier functions** as a multi-turn robustness defense family (positioning GALILEO as complementary measurement).
- [ ] Method idea: define “robustness invariants” for GALILEO tasks (e.g., truthfulness/consistency sets) and measure how often trajectories leave/re-enter them.

## Quotes / details to potentially cite

- “...multi-turn jailbreaks that exploit contextual drift over multiple interactions...” (abstract)
- “...a safety steering framework grounded in safe control theory, ensuring invariant safety in multi-turn dialogues.” (abstract)
- “...introduces a novel neural barrier function (NBF) to detect and filter harmful queries emerging from evolving contexts proactively.” (abstract)
