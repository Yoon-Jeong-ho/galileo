# Sycophancy Is Not One Thing: Causal Separation of Sycophantic Behaviors in LLMs

- Year: 2025
- Venue: arXiv
- Authors: Daniel Vennemeyer; Phan Anh Duong; Tiffany Zhan; Tianyu Jiang
- URL: https://arxiv.org/abs/2509.21305
- BibTeX key (if we add it): vennemeyer2025sycophancy
- Tags: sycophancy, mechanistic-interpretability, steering, representation-geometry, evaluation

## One-sentence takeaway

Sycophantic agreement, genuine agreement, and sycophantic praise are encoded as **distinct, linearly separable and independently steerable** directions/subspaces in LLM residual activations, rather than a single “sycophancy” mechanism.

## What problem does it solve?

- Clarifies whether “sycophancy” is a single latent mechanism vs multiple separable behaviors.
- Provides an operational decomposition: **sycophantic agreement (SyA)** vs **genuine agreement (GA)** vs **sycophantic praise (SyPr)**.
- Supports behavior-selective interventions (suppress harmful deference without harming appropriate agreement/helpfulness).

## What is the core method / protocol?

- Build controlled (mostly synthetic) datasets where:
  - user claim is correct vs incorrect (to separate GA vs SyA), and
  - response contains user-directed praise vs not (to create SyPr variants).
- Use a “model knows the answer” filter: only analyze items where a neutral prompt shows the model strongly prefers the ground-truth answer (to avoid confusing ignorance with sycophancy).
- Learn **difference-in-means (DiffMean)** linear directions from residual-stream activations for each behavior (SyA, GA, SyPr).
- Analyze:
  - layerwise separability (AUROC of linear score along direction),
  - geometry between behavior subspaces (cosine similarity / subspace PCs across datasets),
  - causal steering via **activation addition** along the learned directions (including projecting out other directions to test independence).
- Replicate across multiple model families/scales (paper claims consistent structure across families and sizes).

## What are the key metrics?

- AUROC for linear separability of behaviors using DiffMean direction scores.
- Cosine similarity between principal directions (or subspace components) across behaviors and layers (to track “entanglement → separation”).
- Causal effect of activation addition on:
  - sycophantic agreement rate,
  - genuine agreement rate,
  - praise rate,
  with “cross-effects” as a key diagnostic.

## What are the main results?

- **SyA, GA, SyPr are distinct**:
  - SyA and GA are **entangled early** (model mainly represents “agreement” generically), but **diverge in later layers** into distinct directions.
  - SyPr is largely **orthogonal** to agreement-related directions throughout.
- **Independent controllability**:
  - activation additions can amplify/suppress each behavior with minimal interference on the others.
  - the separation holds even when removing other behavior directions (suggesting it’s not an artifact of shared variance).
- **Cross-model consistency**:
  - representational structure appears stable across model families/scales (as reported).

## How is this similar to GALILEO?

- Shared focus on **pressure-driven behavior change** and the need to disentangle superficially similar phenomena.
- Supports GALILEO’s general stance that we should report **decomposed outcomes** (not one “robustness” scalar), because multiple latent mechanisms can underlie “agreement with the user”.

## How is this different from GALILEO?

- This is primarily **mechanistic / representation** work (latent directions + activation additions) on controlled datasets; GALILEO is positioned more as an **evaluation/protocol/metrics** framework for multi-turn drift vs revision (and may target black-box settings).
- Focuses on single-turn (or controlled) behavior labels; less emphasis on long-horizon multi-turn trajectories, survival/time-to-failure, or recovery dynamics.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO is black-box or model-agnostic, it can apply broadly without internal activations.
- GALILEO can emphasize trajectory metrics (time-to-flip, recovery-after-flip, oscillation), which this paper does not foreground.

## Where GALILEO is weaker / needs to improve

- GALILEO risks collapsing “sycophancy” into a single axis; this paper suggests we should at least separate:
  - **agreement with user stance** (potentially harmful when user is wrong), vs
  - **praise/flattery** (a different channel that may co-occur but is separable).

## Action items for GALILEO (experiments / method / writing)

- [ ] Add an explicit **praise/flattery** annotation channel (SyPr-like) alongside stance-agreement outcomes; report them separately.
- [ ] In writing: cite this as evidence that “sycophancy” is not monolithic; motivate decomposed metrics/labels.
- [ ] If we have any white-box slice: test whether our drift/revision factors align with distinct linear directions; check whether “pressure-agreement” vs “evidence-update” occupy different subspaces.
- [ ] When discussing mitigations: argue for **behavior-selective** interventions (avoid “blunt” anti-sycophancy that may harm appropriate agreement).

## Quotes / details to potentially cite

- Abstract-level claim: sycophantic agreement, genuine agreement, and sycophantic praise are encoded along **distinct linear directions**, are **independently steerable**, and the structure is **consistent across model families/scales**.
- Intro-level framing: early layers encode a generic agreement signal; later layers separate genuine vs sycophantic agreement, while praise stays orthogonal.
