# Language Model Fine-Tuning on Scaled Survey Data for Predicting Distributions of Public Opinions

- Year: 2025
- Venue: arXiv (cs.CL)
- Authors: Joseph Suh; Erfan Jahanparast; Suhong Moon; Minwoo Kang; Serina Chang
- URL: https://arxiv.org/abs/2502.16761
- BibTeX key (if we add it): subpopSuh2025
- Tags: opinions, survey, distributional-prediction, fine-tuning, subpopulations, calibration

## One-sentence takeaway

Fine-tuning an LLM on large-scale survey-derived *subpopulation → response-distribution* supervision (SubPOP) substantially improves distributional matching to humans and generalizes to unseen surveys/subpopulations.

## What problem does it solve?

- Prompt-only “steering” of LLMs to mimic subpopulation opinions typically fails to reproduce *response distributions* (not just a single answer), and tends to over-reflect overrepresented demographics.
- Public-opinion researchers want cheap, early-stage estimates of how different subgroups might respond to candidate survey questions (for pilot testing / oversampling decisions), ideally with uncertainty/variance.

## What is the core method / protocol?

- Curate **SubPOP**: 3,362 questions and ~70K *(subpopulation, question) → response distribution* pairs from large surveys.
  - Train split from Pew American Trends Panel (ATP) waves not in OpinionQA.
  - Evaluate generalization to (i) OpinionQA (ATP subset) and (ii) a different survey family (GSS / NORC).
- Cast each example as: input prompt = (subpopulation descriptor + survey question + response options), target = the **empirical distribution** of human responses for that subpopulation.
- Supervised fine-tuning objective encourages the model’s predicted response distribution to match the human distribution (paper reports results in terms of distribution distance).

## What are the key metrics?

- **Wasserstein distance** between model-predicted and human response distributions (lower is better).
- “LLM–human gap” reductions relative to baselines (prompting / prior methods).

## What are the main results?

- Fine-tuning on SubPOP reduces distributional mismatch substantially (reported **~32–46%** reduction vs strong baselines) across subpopulations.
- Generalization holds to:
  - unseen subpopulations,
  - new survey waves/topics,
  - and even a different survey family (institution).

## How is this similar to GALILEO?

- Same underlying concern: *stability/faithfulness of model outputs under conditioning* (here: demographic/ideological conditioning; in GALILEO: multi-turn pressure/persuasion/flip dynamics).
- Uses an explicitly **distributional** target rather than single-point labels—useful framing for “belief state” as a distribution that can drift.

## How is this different from GALILEO?

- Not multi-turn: predicts survey response distributions in a largely single-turn setup (question + subgroup prompt).
- Focuses on *population-level* opinion distribution prediction, not *within-dialogue* robustness under adversarial/persuasive interaction.

## Where GALILEO is stronger / cleaner (if true)

- GALILEO’s scope is interactional robustness (pressure, persuasion, multi-turn trajectories), closer to real deployment failure modes for assistants.
- GALILEO can measure *temporal/turn-by-turn* dynamics (e.g., “time-to-flip”), which SubPOP does not address.

## Where GALILEO is weaker / needs to improve

- GALILEO likely lacks a principled **distributional calibration** view of “stance/belief” (it may score flips as discrete events rather than distributional shifts).
- SubPOP suggests that *prompting-only* steering is limited; if GALILEO relies heavily on prompting to control drift, it may underestimate the benefit of targeted fine-tuning objectives.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a distributional metric variant: treat responses across seeds/paraphrases/decoding stochasticity as an empirical distribution; score drift using Wasserstein distance / Earth Mover’s Distance across turns/conditions.
- [ ] Consider a “SubPOP-style” supervision analogue for GALILEO: train (or simulate) a target distribution over answers under *pressure conditions* and penalize distribution shift when only framing changes.
- [ ] In related work, cite SubPOP as evidence that prompt-only steering underperforms and that *structurally aligned supervision* can improve subgroup-conditional faithfulness.

## Quotes / details to potentially cite

- SubPOP scale claim: “3,362 questions and 70K subpopulation-response pairs” (fine-tuning dataset derived from large surveys).
- Reported gains: “reducing the LLM-human gap by up to 46% compared to baselines” (distributional match improvements).
