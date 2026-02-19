# Belief Dynamics Reveal the Dual Nature of In-Context Learning and Activation Steering

- Year: 2025
- Venue: arXiv
- Authors: Eric Bigelow; Daniel Wurgaft; YingQiao Wang; Noah Goodman; Tomer Ullman; Hidenori Tanaka; Ekdeep Singh Lubana
- URL: https://arxiv.org/abs/2511.00617
- BibTeX key (if we add it): bigelow2025beliefdynamics
- Tags: belief-updating, in-context-learning, activation-steering, bayesian-model, control

## One-sentence takeaway

A unified Bayesian “belief updating” model explains and predicts how both many-shot prompting (evidence accumulation) and activation steering (prior shift) can cause sigmoidal and phase-transition-like behavior changes, including sudden jailbreak-like flips.

## What problem does it solve?

- We have two major families of *inference-time control* techniques for LLMs—prompting/ICL and activation steering—without a shared predictive theory.
- Practitioners often see *sudden* behavioral shifts as context length increases (e.g., many-shot ICL, many-shot jailbreaking), but lack a simple model to predict where the “flip” point will be.

## What is the core method / protocol?

- Propose a Bayesian latent-concept belief model:
  - In-context learning updates beliefs by *accumulating evidence* (likelihood updates from exemplars).
  - Activation steering updates beliefs by shifting *priors* over latent concepts.
- Fit / validate the resulting closed-form model on experiments that vary:
  - number of in-context exemplars (“shots”), and
  - steering vector magnitude.
- Key predicted phenomenon: additivity in **log-belief space** (prior + evidence) yields “phases” where small intervention changes can trigger large behavioral changes.

## What are the key metrics?

- Behavioral outcome as a function of (shots, steering magnitude), emphasizing:
  - shape of learning curve vs shots (notably **sigmoidal**),
  - horizontal/vertical shifts of the curve under steering,
  - location of “sudden change” / phase boundary that the model predicts.

## What are the main results?

- ICL behavior vs number of exemplars often follows a sigmoidal curve (slow → rapid transition → saturation).
- Activation steering magnitude predictably shifts ICL behavior (consistent with a prior shift).
- Combined interventions behave approximately additively in log-belief, producing distinct regions where behavior is stable vs rapidly changing.
- The fitted model can predict the “critical point” where sudden behavioral change occurs, and the authors position this as a concrete handle on many-shot jailbreaking-style effects.

## How is this similar to GALILEO?

- Same overarching theme: *multi-turn / sequential interaction dynamics* can produce sharp transitions (drift/flip) and we want models/metrics that predict when a system will fail.
- Useful conceptual bridge for GALILEO’s “pressure” and “drift controls”: interpret conversational pressure as shifting priors or injecting evidence.

## How is this different from GALILEO?

- Focuses on inference-time control theory (ICL + activation steering), not on social pressure / sycophancy per se.
- Primary objects are latent concepts and belief updates, not turn-by-turn robustness metrics (e.g., time-to-failure) or user-pressure protocols.
- Experiments are “many-shot control” style; not necessarily adversarial social interaction with explicit pressure manipulations.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO targets *behavioral robustness under social pressure* with explicit controls and multi-turn protocols, it is more directly aligned to real conversational risk modes (sycophancy/persuasion) than a generic control-theory account.
- GALILEO can emphasize externally measurable robustness endpoints (flip time, recovery, refusal consistency) that don’t require assumptions about latent concepts.

## Where GALILEO is weaker / needs to improve

- GALILEO may lack a compact predictive theory linking different “control knobs” (prompting style, pressure intensity, model steering/guardrails) into one equation-level account.
- GALILEO could benefit from explicit modeling of *phase transitions* (critical thresholds) rather than only empirical curves.

## Action items for GALILEO (experiments / method / writing)

- [ ] In the related work, frame “pressure” prompts as potentially operating via **prior shifts** (social cues) and “evidence” prompts as **likelihood updates**; connect to belief-updating language.
- [ ] Add an analysis section looking for **sigmoidal / phase-transition** behavior in GALILEO outcomes vs pressure intensity / number of pressure turns (fit a simple logistic or threshold model).
- [ ] If GALILEO combines interventions (e.g., pressure + recovery prompt), test whether effects are additive in a transformed space (log-odds / logit), mirroring the paper’s “log-belief additivity.”

## Quotes / details to potentially cite

- “...both context- and activation-based interventions impact model behavior by altering its belief in latent concepts: steering operates by changing concept priors, while in-context learning leads to an accumulation of evidence.”
- “...additivity of both interventions in log-belief space... sudden and dramatic behavioral shifts can be induced by slightly changing intervention controls.”
