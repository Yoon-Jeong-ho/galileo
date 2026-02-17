# A Representation Engineering Perspective on the Effectiveness of Multi-Turn Jailbreaks

- Year: 2025
- Venue: arXiv (ICML 2025 submission)
- Authors: Blake Bullwinkel; Mark Russinovich; Ahmed Salem; Santiago Zanella-Beguelin; Daniel Jones; Giorgio Severi; Eugenia Kim; Keegan Hines; Amanda Minnich; Yonatan Zunger; Ram Shankar Siva Kumar
- URL: https://arxiv.org/abs/2507.02956
- BibTeX key (if we add it): Bullwinkel2025RepE_MultiTurnJailbreaks
- Tags: multi-turn, jailbreak, Crescendo, representation-reading, circuit-breakers

## One-sentence takeaway

Multi-turn Crescendo jailbreaks can succeed because the model’s *intermediate representations* of the evolving dialogue stay in a “benign” region (increasingly so with more turns), which helps explain why many single-turn jailbreak defenses (e.g., circuit breakers) fail to generalize to multi-turn settings.

## What problem does it solve?

- Provides a mechanistic/representation-level explanation for why **multi-turn jailbreaks** (specifically Crescendo) remain effective against safety-aligned models and even against defenses tuned for **single-turn** attacks.
- Diagnoses a **generalization gap**: defenses that block obvious single-shot harmful prompts can be bypassed by conversational, incremental escalation.

## What is the core method / protocol?

- Focuses on the **Crescendo** multi-turn jailbreak (gradual escalation using seemingly benign intermediate steps and referencing prior assistant responses).
- Uses **representation reading** (RepE): train simple classifiers (e.g., logistic regression/MLP probes) on hidden representations to distinguish *benign vs harmful* regions using paired prompt–response data.
- Key analysis trick: hold the **final harmful response tokens** fixed, but vary how many prior turns are included as context (k most recent turns), then measure how the representation of those same final-response tokens shifts as k grows.
- Compares:
  - Llama-3-8B-Instruct (standard RLHF-aligned)
  - Llama-3-8B-Instruct-RR (further fine-tuned with RepE “circuit breakers” to resist single-turn jailbreaks)

## What are the key metrics?

- Probe-based “benign vs harmful” classification scores/probabilities on intermediate representations (how “benign-looking” the internal state is).
- Representation similarity / region-of-space analyses across different k (number of turns included).

## What are the main results?

- Safety-aligned LMs often internally represent Crescendo trajectories as **more benign than harmful**, and this effect becomes **stronger with more turns**.
- At each step, Crescendo prompts tend to keep the model in a benign representation region, effectively “walking” the model toward producing disallowed content without triggering defenses.
- Helps explain why **single-turn defenses** like circuit breakers can look strong on single-turn datasets yet be **ineffective** against multi-turn jailbreaks.

## How is this similar to GALILEO?

- Same broad stance: you can learn a lot (and potentially build mitigations) by analyzing **internal representations / trajectories** rather than only surface-form text.
- Highlights the importance of **multi-turn dynamics**: what matters is not just *whether* a model fails, but *how* the state evolves across turns.

## How is this different from GALILEO?

- This paper is centered on **harmful-content jailbreaks** (Crescendo) and the benign/harmful boundary for safety filtering, not (primarily) belief revision vs pressure-driven drift (if that is GALILEO’s focus).
- Mostly **diagnostic/explanatory** (why the attack works / why defenses fail) rather than proposing a full new multi-turn mitigation with strong end-to-end guarantees.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO explicitly separates *evidence-driven updating* from *pressure-only drift* and reports recovery/trajectory metrics, that would be a more decision-relevant evaluation lens than benign/harmful binary regions alone.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently relies on single-turn safety/robustness checks, this paper is evidence we should assume **single-turn success will not transfer** to multi-turn settings.
- If GALILEO lacks representation-trajectory diagnostics, adding them could help explain failures and guide mitigations.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a **multi-turn “gradual escalation”** operator (Crescendo-like) to the evaluation suite; report how robustness changes with number of turns.
- [ ] Include a **representation/trajectory diagnostic** view: track whether internal states remain in a “benign-looking” region even as the conversation approaches failure.
- [ ] In the paper narrative: explicitly call out the **generalization gap** between single-turn defenses and multi-turn attacks, citing this work as mechanistic support.

## Quotes / details to potentially cite

- “Safety-aligned LMs often represent Crescendo responses as more benign than harmful, especially as the number of conversation turns increases.”
- “Crescendo prompts tend to keep model outputs in a ‘benign’ region of representation space… effectively tricking the model into fulfilling harmful requests.”
- “Single-turn jailbreak defenses like circuit breakers are generally ineffective against multi-turn attacks… motivating mitigations that address this generalization gap.”
