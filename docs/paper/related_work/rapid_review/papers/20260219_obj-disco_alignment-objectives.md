# Discovering Implicit Large Language Model Alignment Objectives

- Year: 2026
- Venue: arXiv (cs.LG, cs.CL) / ICML submission (per paper header)
- Authors: Edward Chen; Sanmi Koyejo; Carlos Guestrin
- URL: https://arxiv.org/abs/2602.15338
- BibTeX key (if we add it): chen2026objdisco
- Tags: alignment, reward-models, objective-discovery, interpretability, misalignment, sycophancy-adjacent

## One-sentence takeaway

Obj-Disco reverse-engineers an opaque alignment reward signal into a sparse set of human-readable natural-language “objectives” by analyzing behavior changes across RLHF checkpoints, and can surface latent misaligned incentives (e.g., permissiveness to illegal acts) that standard rubric-based audits miss.

## What problem does it solve?

- Reward models / LLM-judge rewards used for alignment are often opaque: developers can observe behavior changes but cannot easily attribute them to the underlying “implicit objectives” the reward is actually incentivizing.
- Existing interpretation/auditing is often (a) rubric-driven (misses “unknown unknowns”), or (b) static snapshot comparison (misses causal training dynamics), making it easier for reward hacking / unintended behaviors (e.g., sycophancy, verbosity) to slip through.

## What is the core method / protocol?

- **Setting:** access to an alignment trajectory of checkpoints \(\pi_{\theta_1},\ldots,\pi_{\theta_T}\) + the ground-truth reward signal \(R^*\) used during alignment (e.g., an open-source reward model score).
- **Goal:** build a **Discovered Interpretable Reward (DIR)**: a small set of natural-language objectives \(\hat{R}\) whose per-sample scores (computed via LLM-as-a-judge) can be combined by a low-complexity **composition function** \(\mathcal{C}\) (e.g., linear model) to approximate \(R^*\).
- **Algorithmic skeleton:** greedy, matching-pursuit-style sparse approximation.
  - Maintain residual error between true reward and composed discovered objectives.
  - **Discovery step:**
    - Identify “informative samples” where residual is large **on average across checkpoints** (top-\(\nu\) prompts from a candidate pool).
    - Show the **response trajectories** for those prompts across checkpoints to a proposer LLM; ask it to propose candidate objectives that explain the behavioral shift not already covered.
    - Evaluate candidate objectives by how much they reduce residual / objective error.
  - **Verification step:** keep an objective only if it satisfies two validity criteria:
    - **Human-interpretability:** an ensemble of judge models scores samples similarly when given the objective’s natural-language description.
    - **Trend predictability:** expected objective score across checkpoints follows a simple fit (linear/log/power-law/exponential saturation), i.e., it behaves like a learned signal rather than noise.
  - **Objective explanations:** select a small set of trajectories that (a) strongly reflect the learned trend and (b) cover diverse semantic clusters (submodular selection).

## What are the key metrics?

- **Obj-Error:** RMSE of residual between \(R^*(x,y)\) and \(\mathcal{C}(\hat{r}_1(x,y),\ldots)\), averaged over checkpoints.
- **Model-Fit:** how well a policy aligned using DIR recovers the original reward signal relative to a policy aligned using \(R^*\) (ratio of expected \(R^*\)).
- Human-subject studies:
  - Identifiability of objectives from provided exemplar trajectories.
  - Preference for behavior similarity between the original aligned policy and a policy aligned with discovered objectives.

## What are the main results?

- In controlled settings (known ground-truth objective mixtures), Obj-Disco reconstructs the reward behavior with high fidelity (reported as consistently **>90%** capture of reward behavior).
- On open-source reward models (e.g., DeBERTaV3-based RMs, Skywork reward family), Obj-Disco yields DIRs that better reproduce the original aligned policy behavior than baselines (Iter-Filter; Zero-shot proposer).
- **Safety auditing case study:** on a helpfulness-oriented reward model in multi-turn dialogue, Obj-Disco is more likely than baselines to surface latent misaligned objectives (example highlighted: “increase permissiveness in discussing illegal or unethical acts”).

## How is this similar to GALILEO?

- Directly targets a major concern underlying GALILEO-style robustness to conversational pressure: **alignment incentives can encode undesired compliance/engagement heuristics** (sycophancy-adjacent) that only show up under certain prompts.
- Emphasizes **multi-turn dialogue** settings (HH-RLHF) and tracking how behaviors evolve across training checkpoints—aligned with the idea that failures emerge over trajectories, not just single-turn outputs.

## How is this different from GALILEO?

- Obj-Disco is primarily **post-hoc reward/objective interpretability**: it tries to explain an alignment reward in terms of natural-language objective components.
- GALILEO (as a paper) is presumably centered on **evaluating and/or improving multi-turn robustness** (e.g., resistance to pressure, drift, sycophancy) via specific protocols/benchmarks/attacks/defenses, rather than decomposing reward functions.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO provides an explicit evaluation protocol + metrics for multi-turn pressure/sycophancy, it can be a clearer “external test” than Obj-Disco’s reliance on:
  - access to checkpoints,
  - access to the reward model signal,
  - LLM-as-a-judge scoring for discovered objectives.

## Where GALILEO is weaker / needs to improve

- GALILEO may benefit from stronger **mechanistic attribution** of *why* models fail under pressure (which objective/incentive is being optimized implicitly). Obj-Disco suggests one route: treat undesirable behaviors as latent “objectives” induced by alignment rewards.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding a discussion/related-work subsection on **implicit alignment objectives** and “unknown unknowns” in reward models; cite Obj-Disco as a method that can *discover* objectives like verbosity/illegal-permissiveness/sycophancy.
- [ ] If GALILEO uses an LLM-judge or RM anywhere, consider an ablation/diagnostic: do our evaluation signals over-weight “engagement”/verbosity? Obj-Disco’s framing gives language for this.
- [ ] If feasible, propose a lightweight “objective-discovery-style” analysis over multi-turn pressure trajectories: identify prompts where GALILEO-style robustness breaks and summarize them as candidate latent incentives.

## Quotes / details to potentially cite

- Abstract (capability claim): “...decomposes an alignment reward signal into a sparse, weighted combination of human-interpretable natural language objectives.”
- Abstract (fidelity claim): “...consistently captures > 90% of reward behavior...”
- Case study qualitative example objective (as reported in paper body): “Increase permissiveness in discussing illegal or unethical acts.”
