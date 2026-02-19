# A unified framework for evaluating the robustness of machine-learning interpretability for prospect risking

- Year: 2025 (journal); 2026 (arXiv v1)
- Venue: Geophysics 90(3): IM103–IM118 (accepted); arXiv cs.LG
- Authors: Prithwijit Chowdhury; Ahmad Mustafa; Mohit Prabhushankar; Ghassan AlRegib
- URL: https://arxiv.org/abs/2602.14430
- BibTeX key (if we add it): chowdhury2025unified
- Tags: interpretability, robustness, XAI, necessity-sufficiency, counterfactuals, tabular

## One-sentence takeaway

They propose a model-agnostic way to quantify *necessity* and *sufficiency* of tabular features via forward perturbation “pseudo-counterfactuals”, and use it to stress-test whether LIME/SHAP feature rankings are robust/causal enough for hydrocarbon prospect-risking models.

## What problem does it solve?

- Feature-attribution explainers (notably LIME and SHAP) can disagree on “important features”, especially in high-dimensional tabular data.
- Practitioners lack a concrete, causal-flavored *evaluation* to judge whether an attribution explanation is trustworthy (and whether a model+explainer pairing is reliable under noisy/erroneous inputs).

## What is the core method / protocol?

- Define/operationalize *necessity* (roughly: if changing a feature’s value frequently flips the prediction, that feature value is necessary for the original outcome) and *sufficiency* (roughly: if setting a feature to a reference value often drives examples to the reference outcome, that feature value is sufficient).
- Instead of generating “valid/unique” counterfactuals (e.g., DiCE/Wachter) in sparse high-D spaces, they:
  - generate many *forward* perturbations of a single feature while holding others fixed (pseudo-counterfactuals),
  - query the trained classifier directly (no surrogate),
  - estimate necessity/sufficiency as fractions over these interventions, and average across many instances.
- Use aggregated (“global”) necessity and sufficiency scores per feature group to sanity-check against domain knowledge.
- Robustness evaluation for LIME/SHAP:
  - take top-k features ranked by LIME/SHAP across many test points,
  - compute necessity/sufficiency for those features at each rank,
  - check whether higher-ranked features are *also* more necessary/sufficient (their robustness hypothesis).

## What are the key metrics?

- Necessity score (fraction of single-feature perturbations that change the model output), averaged over N instances.
- Sufficiency score (fraction of interventions to a reference value that yield the reference output), averaged over R references.
- “Global” importance via aggregating local necessity/sufficiency magnitudes across the dataset.
- Standard predictive performance (train/validation accuracy) for the underlying classifiers.

## What are the main results?

- On proprietary hydrocarbon prospect-risking tabular data (DHI-style attributes; 35 attributes, also grouped into top/high-level features), they evaluate several sklearn classifiers (logistic regression, Gaussian NB, random forest, and a weighted voting classifier).
- They find cases where LIME/SHAP top-ranked features are *not* proportionately necessary and/or sufficient, i.e., the attribution ranking does not monotonically align with necessity/sufficiency.
- Robustness depends on model+explainer pairing (e.g., some pairings appear more “robust” under their test than others).

## How is this similar to GALILEO?

- If GALILEO needs a principled way to evaluate explanations/feature attributions, this paper offers a concrete evaluation protocol based on interventions (necessity/sufficiency) rather than only agreement or stability.
- Emphasizes *robustness of explanations* (not just accuracy), including behavior under noisy/erroneous data.

## How is this different from GALILEO?

- Domain focus is hydrocarbon prospect risking (structured, proprietary tabular features) rather than a general benchmark suite.
- Their “counterfactuals” are simple perturbation-based pseudo-counterfactuals and explicitly assume (or at least operationalize) feature-wise interventions without a full causal model.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO uses causal graphs / realistic intervention constraints / distribution-aware counterfactual generation, it can address the plausibility/feature-dependence limitations of single-feature perturbations.
- If GALILEO is domain-agnostic with public datasets, it can be more reproducible than this work (dataset here is proprietary).

## Where GALILEO is weaker / needs to improve

- If GALILEO currently lacks necessity/sufficiency-style diagnostics, this paper is a strong pointer for adding causal-flavored evaluation criteria beyond attribution agreement.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding necessity/sufficiency (or close variants) as evaluation axes for explanation quality (especially for tabular / high-dimensional settings).
- [ ] In related work, cite this as evidence that LIME/SHAP rankings can be misaligned with necessity/sufficiency, motivating robustness evaluation.
- [ ] If using perturbation tests, be explicit about feature dependence / plausibility constraints and how they differ from their simpler interventions.

## Quotes / details to potentially cite

- Motivation: LIME and SHAP can yield different feature rankings because “importance” definitions differ; the paper argues necessity/sufficiency provide more theoretically grounded criteria.
- Setup: compare multiple classifiers (LR, GNB, RF, weighted voting) and test robustness of LIME/SHAP rankings via necessity/sufficiency computed from perturbation-based pseudo-counterfactuals.
