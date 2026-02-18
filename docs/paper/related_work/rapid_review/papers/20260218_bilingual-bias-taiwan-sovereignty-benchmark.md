# Bilingual Bias in Large Language Models: A Taiwan Sovereignty Benchmark Study

- Year: 2026
- Venue: arXiv
- Authors: Ju-Chun Ko
- URL: https://arxiv.org/abs/2602.06371
- BibTeX key (if we add it): ko2026bilingualbias
- Tags: bias, multilingual, consistency, evaluation, political-sensitivity

## One-sentence takeaway

A bilingual (Chinese vs English) benchmark on Taiwan-sovereignty questions shows most LLMs change stance/behavior across languages, motivating explicit *cross-language consistency* metrics (LBS, QAC) and audits for politically sensitive topics.

## What problem does it solve?

- Multilingual deployments can hide *language-conditioned* failures: the same model may answer a politically sensitive question differently (or refuse) depending on whether it is asked in Chinese vs English.
- Existing safety/bias evaluations often treat “the model” as language-invariant, which can miss these inconsistencies.

## What is the core method / protocol?

- Construct a small benchmark of Taiwan sovereignty / political-status questions.
- Query 17 LLMs in two languages (Chinese and English) and score responses (0–10) per language.
- Introduce metrics intended to capture (i) language-dependent stance changes and (ii) consistency adjusted for output quality:
  - Language Bias Score (LBS)
  - Quality-Adjusted Consistency (QAC)
- Qualitative categorization of failure modes (e.g., refusals, narrative propagation, censorship-like behavior).

## What are the key metrics?

- Per-language score (0–10) on the benchmark.
- LBS: quantifies disparity in stance/answers across languages.
- QAC: consistency measure that discounts “consistent” but low-quality outputs (e.g., consistent refusal/censorship).

## What are the main results?

- 15/17 models show measurable language bias (different substantive stance / behavior across languages).
- Chinese-origin models reportedly show severe failures (frequent refusal and/or pro-CCP narrative propagation).
- Only GPT-4o Mini is reported as 10/10 in both languages in their setup; even strong flagship models show headroom.
- Some Western models do worse in Chinese than English, suggesting that training-data effects (or system-level filtering) can induce language-conditional degradation.

## How is this similar to GALILEO?

- Shares the core theme that *robustness must be evaluated as a protocol* (conditions/perturbations), not as a single scalar model property.
- Provides a concrete example of “conditioned failure”: language is a perturbation axis analogous to the kinds of context/pressure perturbations GALILEO cares about.
- Uses metrics intended to separate “consistent-but-bad” behavior from genuine robustness (conceptually similar to needing quality-aware stability metrics).

## How is this different from GALILEO?

- Focused on geopolitical/political-sensitivity and multilingual consistency, not multi-turn agent robustness, drift, or recovery.
- Benchmark appears narrow-domain and stance-labeled; GALILEO is more about behavioral stability under interaction dynamics (multi-turn) and interventions.

## Where GALILEO is stronger / cleaner (if true)

- GALILEO-style evaluations can generalize the *axis sweep* idea beyond language (e.g., persona, social pressure, tool feedback) with clearer control conditions.
- Likely stronger at multi-turn measurement (time-to-failure / survival-style) than a one-shot bilingual comparison.

## Where GALILEO is weaker / needs to improve

- If GALILEO doesn’t explicitly include a multilingual axis, it may miss language-conditioned failures that matter in real deployments.
- Need guardrails against “consistent refusal” being scored as stability unless quality-adjusted (the QAC motivation).

## Action items for GALILEO (experiments / method / writing)

- [ ] Add “language” as a perturbation axis for at least a small subset of prompts (e.g., EN/KR/ZH) and report cross-language stability.
- [ ] Include a quality-adjusted stability metric: distinguish (a) consistent correctness from (b) consistently low-information refusal.
- [ ] In related work, cite as an example of *consistency auditing across conditioning variables* for sensitive topics.

## Quotes / details to potentially cite

- “language bias — the phenomenon where the same model produces substantively different political stances depending on the query language.” (abstract)
- Introduces metrics: Language Bias Score (LBS) and Quality-Adjusted Consistency (QAC). (abstract)
