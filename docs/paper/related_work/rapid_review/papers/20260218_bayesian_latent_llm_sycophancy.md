# A Bayesian-Latent Model of Large Language Model Sycophancy

- Year: 2025
- Venue: TechRxiv (preprint); also appears as a journal article (International Journal of Information Technology, Springer)
- Authors: P. P. Ray
- URL: https://www.techrxiv.org/users/913189/articles/1293606-a-bayesian-latent-model-of-large-language-model-sycophancy
- BibTeX key (if we add it): Ray2025BayesianLatentSycophancy
- Tags: sycophancy, measurement, bayesian, latent-variable, evaluation-metrics

## One-sentence takeaway
Proposes a Bayesian latent-variable model that treats “sycophancy” as an unobserved per-(prompt, model, user-cue, style) score and derives posterior-based diagnostics (flip rates, directionality, susceptibility) mostly demonstrated via simulation.

## What problem does it solve?
- Gives a formal probabilistic framework to *quantify* and *decompose* sycophantic behavior (agreement-driven response changes) rather than relying on ad-hoc counting alone.
- Separates (i) baseline accuracy, (ii) contextual cue/style effects, and (iii) model-specific susceptibility, while providing uncertainty (posterior intervals).

## What is the core method / protocol?
- Defines baseline correctness per prompt and model: \(\hat{y}^{(0)}_{i,m}\in\{0,1\}\).
- Defines cued correctness under user cue \(u\) and prompt style \(p\): \(\hat{y}^{(1)}_{i,m,u,p}\in\{0,1\}\).
- Defines a flip indicator: \(\Delta_{i,m,u,p}=\hat{y}^{(1)}_{i,m,u,p}-\hat{y}^{(0)}_{i,m}\in\{-1,0,+1\}\) (regressive, none, progressive).
- Introduces a latent “sycophancy score” per configuration: \(S_{i,m,u,p}\sim\mathcal{N}(0,\sigma_S^2)\).
- Uses a generative logistic model for cued correctness with a linear predictor:
  - Feature vector \(x_{i,m,u,p}\) includes intercept + one-hots for model, prompt domain, cue, and style.
  - Log-odds: \(\eta_{i,m,u,p}=x_{i,m,u,p}^\top\beta + \gamma_m\,S_{i,m,u,p}\).
- Places weakly-informative priors (Normal, HalfNormal, HalfCauchy) and performs posterior inference (described as MCMC).

## What are the key metrics?
- Overall “sycophancy rate” (fraction of configurations that flip): \(\widehat{\pi}=\Pr(\Delta\neq 0)\) estimated by counting.
- Directionality among flips: \(\widehat{\pi}_+\) vs \(\widehat{\pi}_-\) (progressive vs regressive shares).
- Average latent magnitude: \(\mathbb{E}[|S|]\) (interpreted as typical “agreement pull”).
- Model susceptibility: posterior mean (or estimate) of \(\gamma_m\).

## What are the main results?
- Primarily simulation-based: shows how varying \(\sigma_S\) (latent variability) and \(\gamma\) (susceptibility) changes flip-related metrics.
- Emphasizes that Bayesian formulation yields posterior distributions (uncertainty quantification) rather than point estimates.

## How is this similar to GALILEO?
- Shared goal: measure and characterize undesirable behavior shifts under contextual pressure (user cue / framing), rather than only evaluating raw accuracy.
- Provides a vocabulary (flip types; susceptibility) that could map onto GALILEO’s robustness/behavioral-change analyses.

## How is this different from GALILEO?
- Focus is a *statistical latent-variable model* and derived metrics; not a benchmark suite or an end-to-end evaluation pipeline.
- Demonstration is largely simulation-driven (at least in the accessible sections), with limited evidence of large-scale real interaction datasets.

## Where GALILEO is stronger / cleaner (if true)
- If GALILEO uses standardized datasets / controlled protocols, it may provide more externally valid empirical grounding than this paper’s simulations.
- GALILEO likely has clearer operationalization of tasks, prompts, and reproducible measurement code paths (depending on our implementation).

## Where GALILEO is weaker / needs to improve
- If GALILEO currently reports mostly point estimates / counts, it could benefit from Bayesian uncertainty and hierarchical pooling ideas here.
- If GALILEO lacks a model-level “susceptibility” parameterization, \(\gamma_m\)-style factors could be a useful abstraction.

## Action items for GALILEO (experiments / method / writing)
- [ ] Consider adding “progressive vs regressive flips” as a standard decomposition in results/plots.
- [ ] Consider reporting uncertainty (e.g., bootstrap or Bayesian credible intervals) for flip rates and other sycophancy metrics.
- [ ] If we have repeated measures across prompts/users/styles, consider a hierarchical model to estimate model susceptibility analogs to \(\gamma_m\).

## Quotes / details to potentially cite
- Introduces latent sycophancy score: \(S_{i,m,u,p}\) (Gaussian latent variable) to quantify agreement pull.
- Defines flip indicator: \(\Delta_{i,m,u,p}\in\{-1,0,1\}\) to distinguish progressive vs regressive flips.
- Logistic formulation: \(\eta=x^\top\beta + \gamma_m S\) with posterior inference to quantify uncertainty.

(Accessible text source for this review: Springer page for DOI 10.1007/s41870-025-02718-3; the TechRxiv page was not directly fetchable due to access restrictions.)
