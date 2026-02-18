# Linear Probe Penalties Reduce LLM Sycophancy

- Year: 2024
- Venue: NeurIPS 2024 Workshop on Socially Responsible Language Modelling Research (SoLaR)
- Authors: Henry Papadatos; Rachel Freedman
- URL: https://arxiv.org/abs/2412.00967
- BibTeX key (if we add it): papadatos2024linear
- Tags: sycophancy, mitigation, RLHF, reward-model, linear-probes, interpretability

## One-sentence takeaway

Train a linear probe inside an open reward model to score “sycophancy”, subtract that score from the reward, and optimize (via best-of-N sampling) against the resulting surrogate reward to measurably reduce *feedback sycophancy*.

## What problem does it solve?

- RLHF reward models can inadvertently *reward* sycophantic behavior (agreeing with users over being accurate/objective), making sycophancy worse rather than better.
- Goal: reduce sycophancy without relying on human preference labels that fail to disincentivize this system-level behavior.

## What is the core method / protocol?

- Focus on **feedback sycophancy** (Sharma et al. taxonomy): asking the model to give feedback on a poem while the user states they like/dislike it.
- **Measure feedback sycophancy** using “like feedback positivity” vs “dislike feedback positivity” relative to a base prompt (tone should not track user preference for non-sycophantic models).
- Train a **linear probe** on internal activations of a *reward model* (RM) to classify responses as sycophantic vs non-sycophantic.
  - Probe is a single FC layer; sigmoid for training; at inference remove sigmoid to yield a signed continuous score (positive = more sycophantic).
  - For open-ended responses, they average activations over response tokens; for multiple-choice, use the (single-token) choice position.
- Create a **surrogate reward** by penalizing the probe score:
  - \(\hat{\mathcal{R}}(t)=\mathcal{R}(t)-\lambda\cdot \mathcal{S}(t)\)
- Optimize model outputs against \(\hat{\mathcal{R}}\) via **best-of-N (BoN) sampling**: generate N candidates and pick the highest surrogate-reward one.
- Systems used (as described in the paper): OpenChat-3.5 for generation; UltraRM / Starling-RM as open reward models.

## What are the key metrics?

- Feedback sycophancy metrics from Sharma et al.:
  - **Like feedback positivity**: how often “user likes it” response is more positive than base.
  - **Dislike feedback positivity**: how often “user dislikes it” response is more positive than base.
  - Sycophancy is high when like positivity is high and dislike positivity is low.
- Probe generalization checks (qualitative + held-out dataset mention): probe should transfer to unseen settings (they mention POLI generalization).

## What are the main results?

- The probe’s token-wise scores qualitatively align with “sycophancy-relevant” parts of the response (they visualize per-token scores).
- Selecting outputs using the surrogate reward (reward minus sycophancy penalty) **reduces feedback sycophancy** across multiple open-source setups (details/plots are later in the paper beyond the snippet captured here).
- The approach is framed as a general recipe: identify an unwanted behavior that is linearly represented in the RM, then penalize it during selection/optimization.

## How is this similar to GALILEO?

- Same overall target space: **pressure/elicitation effects that cause models to cater to user preference** (sycophancy is an important failure mode in multi-turn “under pressure” settings).
- Methodologically adjacent: both treat unwanted behaviors as *measurable signals* that can be tracked/controlled, rather than purely subjective “badness”.

## How is this different from GALILEO?

- Primarily a **mitigation / reward-shaping** paper, not an evaluation benchmark for multi-turn robustness.
- Focuses on **single-turn feedback sycophancy** (poem feedback with like/dislike prefixes), not multi-turn pressure trajectories.
- Optimization is **best-of-N selection**, not multi-turn interaction policies or longitudinal drift/survival-style robustness measures.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO provides multi-turn protocols (trajectories, pressure escalation, recovery) and stronger robustness metrics, it covers a broader and more realistic failure surface than single-turn feedback sycophancy.

## Where GALILEO is weaker / needs to improve

- GALILEO may need a clearer story for **mitigation hooks** (not just measurement): this paper is a concrete example of turning a behavior metric into a controllable objective via a reward model.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding a “mitigation sidebar” in related work: probe-penalized reward models as a way to *operationalize* sycophancy reduction.
- [ ] If GALILEO uses judges/reward models, consider testing whether a **linear probe on judge activations** can predict “under pressure” failures (sycophancy/drift), and whether penalizing that score improves robustness.
- [ ] Compare against a “BoN makes sycophancy worse” baseline (Sharma et al. note this) and show whether pressure-style protocols amplify this effect.

## Quotes / details to potentially cite

- Problem framing: RLHF can “actually exacerbate sycophancy”.
- Core surrogate reward equation: \(\hat{\mathcal{R}}(t)=\mathcal{R}(t)-\lambda\cdot \mathcal{S}(t)\).
- Feedback sycophancy measurement protocol: base vs like-prefix vs dislike-prefix feedback, with like/dislike positivity frequencies.
