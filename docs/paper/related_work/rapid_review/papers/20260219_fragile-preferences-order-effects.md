# Fragile Preferences: A Deep Dive Into Order Effects in Large Language Models

- Year: 2025
- Venue: arXiv
- Authors: Haonan Yin, Shai Vardi, Vidyanand Choudhary
- URL: https://arxiv.org/abs/2506.14092
- BibTeX key (if we add it): yin2025fragilepreferences
- Tags: order-effects, positional-bias, evaluation-robustness, preference-comparisons

## One-sentence takeaway

LLMs show strong, systematic *position/order effects* in comparative judgments (including quality-dependent primacy/recency shifts), which can flip choices to strictly inferior options—suggesting evaluation/decision pipelines must randomize and diagnose “fragile” preferences rather than trust a single ordering.

## What problem does it solve?

- Identifies and characterizes *order/position biases* when LLMs are used to choose among options (pairwise and “triplewise” comparisons), especially in high-stakes decision-support settings.
- Separates cases where an LLM’s preference is stable vs. where it is effectively a tie that gets broken by presentation order.

## What is the core method / protocol?

- Empirical study across multiple LLMs and two domains:
  - Resume comparisons (applied/high-stakes)
  - Color selection (control setting to isolate position effects)
- Analyzes systematic patterns:
  - Quality-dependent shift: when options are all high quality → favor first option; when lower quality → favor later options.
  - Centrality bias in triplewise comparisons (middle option favored).
  - “Name bias” (certain names favored even controlling for demographic signals; per abstract).
- Proposes an extension of a rational choice framing to categorize pairwise preferences:
  - robust vs fragile vs indifferent (to distinguish genuine preference distortion from superficial tie-breaking).
- Mitigation ideas (per abstract): targeted strategies including using temperature to recover underlying preferences.

## What are the key metrics?

- Strength of order effects (preference reversal rate under permutation / position swap).
- Frequency of selecting strictly inferior options due to ordering.
- Comparative magnitude of position bias vs other biases (they report position bias often stronger than gender bias).
- Robust/fragile/indifferent classification rates under the rational-choice-inspired framework.

## What are the main results?

- Order effects are strong and consistent across models and domains.
- Position effects can cause choices that are strictly worse (not just random tie-breaking).
- Two additional biases highlighted (per abstract):
  - centrality bias (middle option in triplewise)
  - name bias
- Position biases are typically stronger than gender biases in their experiments.
- Mitigation: suggests temperature-based approach can help “recover” underlying preferences when order distorts behavior.

## How is this similar to GALILEO?

- Both highlight that model behavior in interactive / decision-like settings can be *unstable under seemingly irrelevant perturbations* (here: ordering; in GALILEO’s space: dialogue trajectory / user pressure / multi-turn context).
- Offers language and framing (“robust vs fragile”) that maps well onto GALILEO’s interest in *robustness vs susceptibility*.

## How is this different from GALILEO?

- Focuses on comparative-choice order effects (pairwise/triplewise option ordering), not user-assistant multi-turn conversational influence.
- Evaluates decision-support style tasks rather than explicit persuasion/sycophancy dynamics.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO evaluates multi-turn conversational susceptibility, it can capture *trajectory-dependent* effects beyond static ordering.
- GALILEO may be able to separate *social influence* (agreement pressure) from generic positional artifacts.

## Where GALILEO is weaker / needs to improve

- If GALILEO uses any pairwise/triplewise comparisons (e.g., judge models, preference elicitation, ranking responses), it may currently under-control for order effects; this paper suggests that can be a dominant confound.

## Action items for GALILEO (experiments / method / writing)

- [ ] If any evaluation includes A/B comparisons (judge chooses between responses), **randomize order** and report variance / reversal rate.
- [ ] Add a simple diagnostic: run each comparison with swapped order and compute a “fragility” score (reversal fraction).
- [ ] Consider adopting terminology: robust vs fragile vs indifferent preferences to discuss susceptibility to nuisance factors.
- [ ] If using multi-option settings, check for **centrality bias** (middle choice) explicitly.
- [ ] (If relevant) Try temperature sweeps / sampling-based aggregation to reduce order-driven distortions.

## Quotes / details to potentially cite

- Abstract (order effect characterization + framing): “Using this framework, we show that order effects can lead models to select strictly inferior options, and that position biases are typically stronger than gender biases.”
- Abstract (quality-dependent shift): “when all options are high quality, models favor the first option, but when quality is lower, they favor later options.”
