# A Hybrid Theory and Data-driven Approach to Persuasion Detection with Large Language Models

- Year: 2025
- Venue: arXiv; ICWSM Workshop Proceedings (per arXiv page)
- Authors: Gia Bao Hoang et al. (see arXiv for full list)
- URL: https://arxiv.org/abs/2511.22109
- BibTeX key (if we add it): hoang2025hybridpersuasion (tentative)
- Tags: persuasion, detection, belief-change, features, random-forest, llm-annotation

## One-sentence takeaway

The paper uses LLM-generated ratings for psychologically-motivated features to train a simple classifier (random forest) that predicts whether an online message will cause belief change, highlighting **epistemic emotion** and **willingness to share** as top predictors.

## What problem does it solve?

- Traditional belief revision / persuasion models are based on face-to-face settings; online text discourse needs scalable modeling.
- Builds a prediction model for **successful persuasion / belief change** using theory-driven features.

## Core method (from abstract)

- Use an LLM to produce **ratings** of features studied in psychological experiments.
- Train a **random forest classifier** to predict belief change outcomes.
- Among eight tested features, top predictors include:
  - epistemic emotion
  - willingness to share

## Relevance to GALILEO

- If GALILEO discusses **persuasion / susceptibility / user alignment drift**, this provides feature-based framing and candidate signals.
- Could motivate adding **feature probes** or **auxiliary supervision** for persuasion-related robustness evaluations.

## Potential citation hooks (from abstract)

- “Hybrid approach… LLM generated ratings… random forest classification model… predicts whether a message will result in belief change.”
- “Epistemic emotion” and “willingness to share” are top predictors.

## Action items for GALILEO

- [ ] Decide whether to include as related work for **persuasion detection** (vs generation) and measurement.
- [ ] Consider whether GALILEO evaluations need a persuasion-detection baseline / features.
- [ ] If we mention persuasion dynamics, cite these two features as interpretable predictors.
