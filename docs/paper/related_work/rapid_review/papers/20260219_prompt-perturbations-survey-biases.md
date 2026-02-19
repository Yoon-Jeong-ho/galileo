# Prompt Perturbations Reveal Human-Like Biases in Large Language Model Survey Responses

- Year: 2025
- Venue: arXiv (cs.CL, cs.AI, cs.CY)
- Authors: Jens Rupprecht, Georg Ahnert, Markus Strohmaier
- URL: https://arxiv.org/abs/2507.07188
- BibTeX key (if we add it): rupprecht2025promptperturbations
- Tags: response-bias, survey-methodology, robustness, prompt-perturbation, recency-bias, synthetic-survey

## One-sentence takeaway

LLMs used as “synthetic survey respondents” are highly sensitive to prompt/option perturbations and consistently show strong recency bias (over-selecting the last option), implying robustness checks and careful instrument design are mandatory.

## What problem does it solve?

- Quantifies how unreliable/sensitive LLM-generated closed-ended survey responses can be under realistic variations in question wording and answer-option formatting.
- Tests whether LLMs exhibit classic human survey response biases (e.g., primacy/recency, central tendency, opinion floating) when presented with normative survey items.

## What is the core method / protocol?

- Uses normative/value-oriented items from World Values Survey (WVS) Wave 7.
- Selects 62 questions (stratified across 10 thematic categories; excludes sociodemographics).
- Defines 10 perturbations applied to either answer options (bias-inducing) or question text (non-bias/noise/semantic variants), plus an interaction case.
  - Examples mentioned: reversed response order; missing refusal (“don’t know”); odd/even scale transformation (introduce a middle/neutral); priming suffix; typos; synonyms; paraphrasing; combined perturbations.
- Runs 25 sampled “interviews” per (question, perturbation, model) across 9 instruction-tuned LLMs (mix of proprietary + open).
- Compares perturbed-response distributions to the original prompt baseline using distributional metrics (entropy, KL divergence) and targeted bias tests (first vs last option; shifts toward center).

## What are the key metrics?

- Distribution shift vs baseline: entropy and KL divergence.
- Response-order bias: relative frequency of selecting first vs last option under order manipulations.
- “Opinion floating” / “central tendency”: movement of mass toward central categories when refusal/middle options are removed/added.

## What are the main results?

- All tested models exhibit a consistent recency bias (favoring the last-presented option), sometimes extremely strongly (reported up to ~20x over-selection).
- Larger models are generally more robust (smaller distribution shifts), but none are fully robust.
- Semantic perturbations (paraphrasing/synonyms) and combined perturbations still noticeably change response distributions.
- Overall implication: synthetic survey outputs can be an artifact of prompt formatting rather than stable “opinions.”

## How is this similar to GALILEO?

- Directly aligns with concerns about robustness / sensitivity of LLM behavior under small prompt changes.
- Provides a concrete perturbation framework and evaluation recipe for “distribution stability” rather than accuracy.
- Highlights a specific, repeatable failure mode (recency bias) that can contaminate any pipeline relying on multiple-choice/ordinal outputs.

## How is this different from GALILEO?

- Focuses on normative survey Q&A (no ground-truth labels), emphasizing robustness and bias patterns rather than task performance.
- Evaluation is distributional over repeated samples (25 runs per condition), not single-run correctness.
- Uses survey-methodology constructs (primacy/recency/opinion floating) as the main lens.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO targets objective tasks or has explicit correctness signals, it can separate “format sensitivity” from genuine performance errors more cleanly.
- If GALILEO already standardizes option presentation / randomizes order / uses calibration, it can mitigate the headline issue (recency bias).

## Where GALILEO is weaker / needs to improve

- If GALILEO uses fixed answer-option ordering (or any ordinal multiple-choice interface), it may inherit the same recency bias artifact.
- If GALILEO reports results from a single prompt template, it may be overconfident without perturbation-based sensitivity analysis.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a “prompt perturbation robustness” section/appendix: test paraphrase + option-order reversal + typo noise for any key multiple-choice components.
- [ ] Always randomize or counterbalance answer-option order (and report this), especially for ordinal scales.
- [ ] If using refusal/unknown options, explicitly test with/without them and quantify distribution shifts.
- [ ] Consider reporting KL divergence / entropy shift vs a canonical prompt as a stability metric.

## Quotes / details to potentially cite

- Abstract-level claim: “all tested models exhibit a consistent recency bias, disproportionately favoring the last-presented answer option” and that perturbations produce large distribution shifts even for larger models.
- Scale: 9 models, 62 WVS questions, 10 perturbations, 25 repeats → 167k+ simulated interviews.
