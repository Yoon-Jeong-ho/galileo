# Robustness of Prompting: Enhancing Robustness of Large Language Models Against Prompting Attacks (RoP)

- Year: 2025
- Venue: arXiv
- Authors: Lin Mu et al. (see arXiv)
- URL: https://arxiv.org/abs/2506.03627
- BibTeX key (if we add it): Mu2025RoP
- Tags: robustness, prompt-perturbation, prompt-attacks, error-correction, guidance

## One-sentence takeaway

RoP is a 2-stage prompting recipe (auto error-correction + guidance prompt) intended to reduce accuracy drops under small input perturbations (typos/character swaps) across reasoning tasks.

## What problem does it solve?

- Prompting-based LLM usage is brittle: minor character-level perturbations in the input (typos, reorderings) can cause large performance degradation.
- Existing prompting methods don’t explicitly target robustness to such perturbations.

## What is the core method / protocol?

- Two-stage prompting strategy:
  - **Stage 1: Error Correction**
    - Generate perturbed variants of the input (adversarial examples via “diverse perturbation methods”).
    - Use these to construct prompts that elicit *automatic correction* of input errors.
  - **Stage 2: Guidance**
    - Given the corrected input, generate an “optimal guidance prompting” that steers the model toward robust inference.
- The paper frames this as a prompting-time defense, not model retraining (as described in the abstract).

## What are the key metrics?

- Accuracy / task performance under:
  - clean inputs
  - adversarially perturbed inputs (character-level / typographical perturbations)
- Reported emphasis: robustness = reduced performance drop between clean vs perturbed.

## What are the main results?

- Across arithmetic, commonsense, and logical reasoning tasks, RoP improves robustness to adversarial perturbations.
- Claim: only minimal degradation relative to clean-input accuracy, i.e., stronger stability under perturbations.

## How is this similar to GALILEO?

- Same broad goal: robustness under adversarial or distributional perturbations to the interaction (here: input text perturbations).
- Uses a **protocol-level intervention** (prompting strategy) rather than changing the base model.

## How is this different from GALILEO?

- Targets **single-input perturbation robustness** (typos/character errors), not multi-turn interaction dynamics (e.g., drift, belief change, pressure, recovery).
- The “attack surface” is mostly **surface-form noise**, not semantic manipulations over time.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO focuses on multi-turn robustness and/or principled metrics (e.g., turn-of-failure / recovery), it covers failure modes RoP doesn’t address.

## Where GALILEO is weaker / needs to improve

- If GALILEO lacks explicit *input-noise stress tests*, RoP suggests a simple additional robustness axis to include.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a “surface perturbation” stress test condition (typos, character swaps) to ensure effects aren’t artifacts of minor formatting noise.
- [ ] In related work, position RoP as a prompting-time robustness defense for character-level perturbations (contrast with GALILEO’s target failure modes).

## Quotes / details to potentially cite

- Abstract: “RoP consists of two stages: Error Correction and Guidance.”
- Abstract: “LLMs … are highly sensitive to input perturbations, such as typographical errors or slight character order errors …”
- Abstract: “RoP significantly improves LLMs' robustness against adversarial perturbations … with only minimal degradation compared to clean input scenarios …”
