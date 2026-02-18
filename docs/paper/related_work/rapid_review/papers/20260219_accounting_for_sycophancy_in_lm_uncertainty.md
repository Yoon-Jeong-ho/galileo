# Accounting for Sycophancy in Language Model Uncertainty Estimation

- Year: 2024
- Venue: arXiv
- Authors: Anthony Sicilia, Mert Inan, Malihe Alikhani
- URL: https://arxiv.org/abs/2410.14746
- BibTeX key (if we add it): sicilia2024accounting
- Tags: sycophancy, uncertainty-estimation, calibration, human-llm-collaboration

## One-sentence takeaway

Sycophancy (agreeing with user suggestions) can systematically distort language-model uncertainty estimates, and conditioning calibration on user-suggestion behavior (SyRoUP) improves correctness-probability prediction.

## What problem does it solve?

- Standard LLM uncertainty estimation (UE) methods typically treat the model’s answer as produced “alone”, but in realistic collaborative settings users provide suggested answers with varying correctness/confidence.
- When the model is sycophantic, it may adopt incorrect user suggestions and (critically) also become over-confident, undermining uncertainty signals meant to help humans intervene.

## What is the core method / protocol?

- Extends UE evaluation to a *suggestion-conditioned* setting:
  - Prompt model with question.
  - Provide a user “suggested answer” (varied by strategy, including correctness and expressed confidence).
  - Evaluate the model’s final answer *and* the uncertainty estimate for that final answer.
- Proposes metrics that generalize “sycophancy bias” to quantify downstream impacts on uncertainty estimation (differences in uncertainty behavior with vs. without suggestions).
- Proposes **SyRoUP**: a modification of Platt scaling used in UE pipelines, where the calibration (mapping derivative/features → probability of correctness) is **conditioned on categorical user-behavior descriptors** (e.g., whether user suggests, and how).

## What are the key metrics?

- Treat UE as probabilistic classification of correctness:
  - **Brier Score** (mean squared error between predicted correctness probability and actual correctness).
  - **Brier Skill Score (BSS)** (variance explained / information gain relative to base rate accuracy).
- Also discusses impacts on *accuracy* (model correctness) under user suggestions.

## What are the main results?

- Sycophancy affects not just accuracy but also the *calibration/quality* of uncertainty estimates in collaborative settings.
- **User confidence** (how confidently the user presents a suggestion) plays an important role in modulating sycophancy’s effect on both accuracy and uncertainty.
- Conditioning calibration on user behavior (**SyRoUP**) better predicts these sycophancy-linked changes than a single global calibration.
- Argument/implication: externalizing **both model uncertainty and user uncertainty** can mitigate sycophancy harms.

## How is this similar to GALILEO?

- Directly targets *reliable uncertainty estimation* for language models under interaction effects (user involvement), aligning with the general goal of making confidence signals trustworthy for downstream decision-making.
- Highlights that “uncertainty estimation” is not purely a model-internal property; the *protocol* (interaction / prompts / additional context) can change calibration.

## How is this different from GALILEO?

- Centers sycophancy specifically (agreement with user-suggested answers) and models user behavior categories; not a generic UE method.
- Uses a relatively lightweight conditional calibration approach (Platt-scaling variant) rather than proposing a new base UE signal.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO’s UE signal is designed to be robust across prompting/decoding variations, it may generalize beyond explicit “user suggestion” categories (where SyRoUP relies on knowing/labeling interaction type).

## Where GALILEO is weaker / needs to improve

- If GALILEO evaluates UE primarily in single-agent prompting, this paper suggests adding a *collaborative* evaluation slice (user suggestions with confidence/correctness) to ensure calibration doesn’t break in realistic usage.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a “user suggestion” perturbation family to UE evaluation: vary (i) suggestion correctness and (ii) expressed user confidence; measure change in calibration (Brier/BSS) and accuracy.
- [ ] Consider conditioning calibration (or at least stratified reporting) on interaction category (no suggestion vs suggestion; high vs low user confidence).
- [ ] In related work, cite this as evidence that interaction artifacts (sycophancy) can *propagate into* uncertainty signals, motivating protocol-aware evaluation.

## Quotes / details to potentially cite

- Abstract framing: sycophancy can make models “over-confident in (incorrect) problem solutions suggested by a user,” and the paper studies sycophancy–UE relationship and proposes SyRoUP.
- Key claim: “user confidence plays a critical role in modulating the effects of sycophancy.”
