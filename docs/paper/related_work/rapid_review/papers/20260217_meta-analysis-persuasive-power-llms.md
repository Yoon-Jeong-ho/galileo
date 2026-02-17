# A meta-analysis of the persuasive power of large language models

- Year: 2025
- Venue: Scientific Reports (Nature) / arXiv preprint
- Authors: Sebastian Maier; Stefan Feuerriegel
- URL: https://www.nature.com/articles/s41598-025-30783-y (also: https://arxiv.org/abs/2512.01431)
- BibTeX key (if we add it): Maier2025MetaAnalysisPersuasiveLLMs
- Tags: persuasion, meta-analysis, human-subjects, evaluation

## One-sentence takeaway

Across 7 eligible experiments (17,422 participants), LLM-generated persuasive messages are *on average* neither more nor less persuasive than human messages (pooled Hedges’ g≈0.02, n.s.), but results vary a lot by context.

## What problem does it solve?

- The empirical literature on “LLMs as persuaders” reports conflicting outcomes (LLMs worse / equal / better than humans), often with incomparable setups.
- This work aggregates evidence with strict inclusion criteria to estimate an overall effect and characterize heterogeneity.

## What is the core method / protocol?

- PRISMA-style systematic review + random-effects meta-analysis.
- Databases searched: Web of Science, ACM DL, arXiv, SSRN, OSF; cutoff May 22, 2025; English; post-Dec 2022.
- Stringent inclusion: direct experimental comparison of LLM-generated vs human-generated persuasive communication, between-subjects / independent observations, and sufficient stats to compute effect sizes.
- Effect size: compute Cohen’s d from reported summary stats → apply small-sample correction → Hedges’ g.
- One effect size per experiment; pooled with REML random-effects.
- Publication-bias checks: Egger’s regression; trim-and-fill; influence / leave-one-out.
- Exploratory moderators: model family, interaction format (interactive vs one-shot), domain.

## What are the key metrics?

- Hedges’ g (LLM vs human persuasive effectiveness)
- Heterogeneity: I², τ², Cochran’s Q
- Publication-bias indicators: Egger’s test; trim-and-fill
- Moderator model explanatory power: R² (between-study variance explained)

## What are the main results?

- Included evidence: 7 eligible studies; 17,422 participants; 12 independent effect-size estimates.
- Overall effect: no significant difference between LLM and human persuasiveness (pooled g = 0.02, p = .530).
- Heterogeneity: substantial (I² ≈ 75.97%), implying strong context dependence.
- Publication bias: Egger’s test suggests possible small-study effects (p = .018), but trim-and-fill imputes no missing studies (authors interpret as low risk).
- Moderator analyses:
  - Single-factor moderator tests (model, conversation design, domain) not statistically significant (likely low power).
  - Joint model explains large share of between-study variance (R² ≈ 81.93%), leaving lower residual heterogeneity (I² ≈ 35.51%).

## How is this similar to GALILEO?

- Same broad target: quantifying susceptibility/persuasive impact and robustness to conversational influence.
- Emphasizes that *protocol details* drive measured effects, aligning with GALILEO’s focus on evaluation design.

## How is this different from GALILEO?

- This is human-subject persuasion effectiveness (LLM → human outcomes), not model-internal belief/consistency under multi-turn pressure.
- Aggregates across studies rather than proposing a new benchmark/protocol.
- Outcomes include attitudes/intentions/compliance/PME, not turn-level stability or recovery metrics.

## Where GALILEO is stronger / cleaner (if true)

- Can define controlled multi-turn perturbation regimes, counterfactual controls, and within-model metrics (e.g., time-to-failure, recovery-to-truth) that are hard to isolate in human-subject settings.
- Can systematically vary interaction factors (pressure, authority, personalization, disclosure) with known ground truth.

## Where GALILEO is weaker / needs to improve

- May not directly translate to real-world “persuasion success” on humans; mapping from model outputs to human behavior is indirect.
- Needs clear framing on how GALILEO metrics relate to human-impact risk (if that’s a motivation).

## Action items for GALILEO (experiments / method / writing)

- [ ] Related-work positioning: cite this as evidence that “LLMs can match humans on persuasion on average” but emphasize the large heterogeneity and dependence on context/protocol.
- [ ] Use it to motivate factorial evaluation: isolate interaction format (one-shot vs interactive), domain, and model family as key contextual drivers.
- [ ] Consider adding a short paragraph distinguishing (a) human-target persuasion effectiveness vs (b) model-target susceptibility/instability under pressure (GALILEO’s niche).

## Quotes / details to potentially cite

- “no significant overall difference in persuasive performance between LLMs and humans (g = 0.02, p = .530)” (abstract/arXiv HTML).
- “substantial heterogeneity across studies (I² = 75.97%)” (abstract/arXiv HTML).
- “combined model … explained … (R² = 81.93%), and residual heterogeneity is low (I² = 35.51%)” (abstract/arXiv HTML).
