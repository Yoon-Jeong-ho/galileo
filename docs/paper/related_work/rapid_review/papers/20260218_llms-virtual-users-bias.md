# LLMs, Virtual Users, and Bias: Predicting Any Survey Question Without Human Data

- Year: 2025
- Venue: 17th International Conference on Machine Learning and Computing (ICMLC), accepted (arXiv)
- Authors: Enzo Sinacola; Arnault Pachot; Thierry Petit
- URL: https://arxiv.org/abs/2503.16498
- BibTeX key (if we add it): sinacola2025llms-virtual-users-bias
- Tags: synthetic-users, survey, public-opinion, demographic-bias, censorship, evaluation

## One-sentence takeaway

LLMs can predict binary survey answers from demographics competitively with Random Forests without any training data, but performance varies across demographic groups and “uncensoring” improves accuracy especially for underrepresented groups.

## What problem does it solve?

- Replace/augment expensive human surveys by generating “virtual respondent” answers to survey questions.
- Benchmark whether LLMs can predict survey outcomes (esp. binary questions) using only demographic profiles.
- Diagnose demographic bias / uneven performance (religion/ethnicity) and study the effect of model censorship/guardrails on accuracy.

## What is the core method / protocol?

- Data: World Values Survey (WVS) Wave 7 (2017–2022), 94,278 individuals, 64 countries; ~35 demographic input features.
- Targets: 196 questions total; focus primarily on 30 binary opinion questions to reduce compute.
- Virtual population prompting: for each respondent (demographic profile), prompt an LLM to output the answer to a given survey question.
- Four phases:
  - Prompt + temperature search (done with Mistral-7B).
  - Model comparison on a fixed test sample (384 individuals) for the 30 binary questions.
  - Grouped evaluation across ethnic and religious groups.
  - Compare censored vs “uncensored” (Dolphin variants) within the same base model family.

## What are the key metrics?

- Accuracy (%) for predicting individuals’ binary responses.
- Stability across repeated runs under different temperature settings (reported as average accuracy per run).
- Group-wise accuracy across religion/ethnicity segments (bias/disparity lens).

## What are the main results?

- Prompting/temperature:
  - Prompt choice has a modest effect; for binary questions, best prompt around ~67.11% avg accuracy (Mistral-7B).
  - Lower temperatures (0 / ~0.001) produce the most consistent results; repeated runs reported ~68.26% avg accuracy per run.
- Model comparison (30 binary questions; 384-person test set):
  - Random Forests trained per-question improves with more training data: ~71.58% (1% train), ~73.40% (5%), up to ~74.93% (95%), outperforming all tested LLMs when enough labeled survey data exists.
  - Top LLMs: Claude 3.5 Sonnet and GPT-4o ~71% average accuracy.
  - Several open models lower (examples reported: Mistral-7B ~63%; Llama-family variants around 60–62%; one Dolphin-Mistral variant reported ~53%).
- Bias / censorship finding (headline):
  - Performance drops for some religious / population groups; “uncensored” Dolphin variants improve predictive accuracy, especially for underrepresented segments where censored models struggle.

## How is this similar to GALILEO?

- Emphasizes evaluation protocols that stress-test model behavior across *subpopulations* rather than only average performance.
- Treats “alignment/guardrail effects” as a first-class variable that can change observed behavior (similar to how safety/refusal dynamics can alter robustness outcomes).
- Uses a multi-phase evaluation design (prompt/temperature calibration → main benchmark → subgroup analysis).

## How is this different from GALILEO?

- Task is primarily single-turn prediction of survey answers from demographics (not multi-turn interaction robustness, belief drift, or adversarial dialogue dynamics).
- Compares against a supervised baseline that is trained per-question (Random Forests), whereas GALILEO’s focus is typically on behavioral robustness metrics/protocols rather than standard supervised prediction.
- “Uncensoring helps accuracy” is specific to survey-style questions; it’s not directly a robustness-to-pressure result.

## Where GALILEO is stronger / cleaner (if true)

- Can capture *trajectory-level* failure modes (when/why a model shifts under conversational pressure) rather than static per-question accuracy.
- Can separate evidence-driven revision from social-pressure-driven acquiescence using controlled multi-turn protocols.

## Where GALILEO is weaker / needs to improve

- If GALILEO does not currently report subgroup slices, it may miss fairness-like disparities (e.g., certain demographic/persona segments failing earlier).
- If GALILEO only uses “aligned” models, results may be confounded by refusal/guardrail behaviors; this paper suggests explicitly comparing aligned vs less-censored variants can change conclusions.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add subgroup/slice reporting (persona/demographic proxies, or topic-sensitive clusters): report robustness metrics per slice, not only averaged.
- [ ] Add an explicit “alignment/guardrails condition” factor: compare base vs safety-tuned vs refusal-heavy variants, and quantify how metrics shift.
- [ ] Consider a simple baseline comparator (non-LLM model or rule-based) for certain tasks, to clarify when “LLM complexity” is necessary.

## Quotes / details to potentially cite

- WVS Wave 7 scale: 94,278 individuals, 64 countries.
- Protocol detail: models tested on a consistent sample of 384 individuals (95% confidence, 5% margin of error rationale given).
- Random Forests accuracy improves with training data size, reaching ~74.93% at 95% train; top LLMs ~71% without additional training data.
- Observation: removing censorship mechanisms improves predictive accuracy, particularly for underrepresented demographic segments.
