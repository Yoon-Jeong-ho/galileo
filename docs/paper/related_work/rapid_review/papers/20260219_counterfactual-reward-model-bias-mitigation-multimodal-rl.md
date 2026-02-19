# Counterfactual Reward Model Training for Bias Mitigation in Multimodal Reinforcement Learning

- Year: 2025
- Venue: arXiv (preprint)
- Authors: Sheryl Mathew
- URL: https://arxiv.org/html/2508.19567v1
- BibTeX key (if we add it): mathew2025counterfactual
- Tags: robustness, drift, reward-model, bias-mitigation, fairness, counterfactual

## One-sentence takeaway

Proposes an RLHF-style reward modeling pipeline that uses counterfactual perturbations plus drift/uncertainty/fairness monitoring to produce a composite “trust” score intended to reduce biased reward signals under multimodal distribution shift.

## What problem does it solve?

- Reward models in multimodal RLHF can amplify latent dataset biases and spurious correlations, especially under temporal drift and confounding, leading to unfair or unreliable policy optimization.
- Existing mitigation approaches framed as “static constraints” may fail when confounders and distribution shift change over time.

## What is the core method / protocol?

- Treat reward modeling as supervised classification (fake vs true news) and then augment it with counterfactual and drift-aware diagnostics.
- Baseline model: CatBoost classifier over mixed features (text-derived embeddings + categorical metadata).
- Add a noise-injection autoencoder (and a “transformer-augmented” encoding block) to detect drift via reconstruction error and representation shift.
- Generate counterfactual variants by perturbing protected attributes (paper is not very specific about which attributes in the extracted text).
- Define a composite Counterfactual Trust Score (CTS) / Trust_t aggregating:
  - Drift metrics (e.g., PSI, JSD, reconstruction-error deltas)
  - Uncertainty proxy (softmax margin between top-1 and top-2)
  - Fairness rule violation rate (per protected attribute)
  - Classification error (and an additional counterfactual consistency penalty term in a later equation)
- Evaluate robustness by injecting synthetic bias into later temporal batches:
  - Subject distribution shift (e.g., over-represent politics)
  - “Framing disturbance” (word swaps affecting framing)
  - Temporal drift (label distribution changes)

## What are the key metrics?

- Accuracy on fake-vs-true classification.
- Drift metrics: Population Stability Index (PSI), Jensen-Shannon Divergence (JSD), autoencoder reconstruction error.
- Uncertainty: probability margin (p_max - p_second_max).
- Fairness: “fairness rule violations” rate (not clearly formalized in the extracted text).
- Composite trust score combining the above with tunable weights.

## What are the main results?

- Reports 89.12% accuracy on the multimodal fake vs true news dataset and claims improvement over baseline reward models.
- Claims reduced spurious correlations / unfair reward assignments and improved sensitivity to bias under synthetic bias injection.
- (From the extracted HTML) results are described at a high level; detailed ablations and statistical testing are not evident in the snippet.

## How is this similar to GALILEO?

- Emphasizes robustness under drift and distribution shift; uses representation-based drift signals and temporal batching.
- Uses composite monitoring signals (drift + uncertainty + fairness) rather than a single scalar objective.
- Frames the problem as avoiding spurious correlations that would otherwise be reinforced by downstream optimization.

## How is this different from GALILEO?

- Centers on a bespoke “trust score” for reward model reliability/fairness; GALILEO’s core contributions may be more about principled objective design / evaluation protocols (depending on our framing), whereas this is an applied pipeline.
- Uses CatBoost + autoencoder + transformer components in an ad-hoc architecture; not clearly tied to a rigorous causal identification strategy beyond counterfactual perturbations.
- Evaluation domain is fake-news classification as a proxy for reward modeling; may be far from our target tasks/settings.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO has a clearer theoretical or empirical protocol (ablations, benchmarks, causal assumptions), we can position it as more principled than a heuristic composite scoring approach.
- Opportunity to argue for clearer definitions of fairness constraints, protected attributes, and counterfactual generation (this paper is vague in the extracted sections).

## Where GALILEO is weaker / needs to improve

- If GALILEO does not explicitly incorporate fairness monitoring or counterfactual consistency checks, this paper is a reminder to address fairness dimensions under drift.
- If we lack an easy-to-communicate “dashboard” metric for reliability under temporal shift, their CTS framing might be a useful narrative device.

## Action items for GALILEO (experiments / method / writing)

- [ ] Related-work positioning: mention composite “trust” scoring approaches that combine drift + uncertainty + fairness as an alternative line to purely constraint-based bias mitigation in reward modeling.
- [ ] Consider adding (or at least discussing) a counterfactual consistency diagnostic: sensitivity of reward/policy outputs to protected-attribute flips.
- [ ] If we already use drift/uncertainty metrics, explicitly note how fairness tracking could be integrated in streaming / temporal settings.

## Quotes / details to potentially cite

- Problem framing (bias amplification in reward models): “reward models can efficiently learn and amplify latent biases within multimodal datasets … imperfect policy optimization through flawed reward signals and decreased fairness.”
- CTS definition (composite trust): counterfactual shifts + reconstruction uncertainty + fairness rule violations + temporal reward shifts, aggregated into a weighted trust score.
- Bias-injection evaluation idea: inject synthetic bias in later temporal batches (subject distribution shift, framing disturbance, label drift) to test robustness.
