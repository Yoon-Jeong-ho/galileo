# Can AI-Generated Persuasion Be Detected? Persuaficial Benchmark and AI vs. Human Linguistic Differences

- Year: 2026
- Venue: arXiv (preprint; under review)
- Authors: Arkadiusz Modzelewski; Paweł Golik; Anna Kołos; Giovanni Da San Martino
- URL: https://arxiv.org/abs/2601.04925
- BibTeX key (if we add it): persuaficial2026modzelewski
- Tags: persuasion, detection, benchmark, multilingual, synthetic-data, interpretability

## One-sentence takeaway

Persuaficial is a ~65k-text, 6-language benchmark of controllably-generated persuasive text showing that *subtle* LLM-generated persuasion is harder for automatic detectors (incl. zero-shot LLM classifiers) than overt persuasion, and providing an interpretable linguistic-feature analysis of AI-vs-human persuasion.

## What problem does it solve?

- Safety/abuse concern: LLMs can generate persuasive content for propaganda/manipulation, so we need to know whether AI-generated persuasion is detectable.
- Gap addressed: prior work studied persuasion detection, but not whether *LLM-generated* persuasion is harder to detect than human persuasion, nor a detailed linguistic comparison between the two.

## What is the core method / protocol?

- Construct a controlled synthetic persuasion dataset (Persuaficial) by taking human texts from 3 persuasion/propaganda datasets and generating AI variants with prompt-controlled approaches.
- Human sources (as described in paper): SemEval 2023 Task 3 persuasion-techniques news dataset; DIPROMATS 2024 Task 1 (X/Twitter diplomat/authority posts); ChangeMyView (Reddit persuasion conversations).
- Four controllable generation approaches (inspired by synthetic misinformation generation):
  - Paraphrasing
  - Rewriting with *subtle* persuasion
  - Rewriting with *intensified* persuasion
  - Open-ended generation from a factual summary of the original
- Multiple LLM generators (paper lists): Gemma 3 27B IT; Llama 3.3 70B; Gemini 2.0 Flash; GPT-4.1 Mini.
- Detection evaluation: treat persuasion detection as binary classification (persuasive vs non-persuasive), run in zero-shot with temperature=0 using multiple LLM “detectors/classifiers” (same set as above).
- Linguistic analysis: compute 196 interpretable/reproducible features using StyloMetrix; compare distributions for human vs LLM-generated persuasive text.

## What are the key metrics?

- F1 score for binary persuasion detection (zero-shot LLM classification).
- Dataset quality checks:
  - Pre-generation check for factual fidelity of summaries used for open-ended generation (human-evaluated).
  - Post-generation multi-criteria validation (factual fidelity, persuasiveness, instruction-following) with unanimous agreement requirement.

## What are the main results?

- Detection difficulty depends strongly on “persuasion strength”:
  - Overt/intensified persuasive generations can be *easier* to detect than human persuasion.
  - Subtle persuasive rewrites consistently *degrade* detector performance (harder than human).
- Persuaficial scale and coverage:
  - ~65k texts total across 6 languages (EN/DE/PL/IT/FR/RU).
  - For each persuasive example: 4 models × 4 approaches = 16 generation configurations.
- Quality evaluation (English sample):
  - Summary factuality accuracy reported around ~91% (conservative definition).
  - Post-generation overall validity around ~88% (conservative “all criteria satisfied”); persuasion-related criteria alone ~97%.
- Provides an interpretable feature-level analysis (StyloMetrix) of how LLM vs human persuasive texts differ (useful for building non-LLM, interpretable detectors).

## How is this similar to GALILEO?

- Both care about robustness/safety under *strategically crafted language* (here: subtle persuasion) and evaluation that separates “hard cases” from easy ones.
- Emphasizes controlled generation regimes to probe model behavior and detector brittleness.

## How is this different from GALILEO?

- Focus is persuasion *detection* (binary classifier) and linguistic stylometry, not interactive multi-turn robustness of an assistant.
- Primary artifact is a multilingual synthetic dataset + analysis, rather than an interactive benchmark/protocol.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO targets interactive, multi-turn settings (pressure, drift, user feedback), it captures dynamics beyond single-text persuasion detection.

## Where GALILEO is weaker / needs to improve

- If GALILEO lacks a clear notion of “subtle vs overt” manipulative pressure, this paper suggests explicitly parameterizing strength and measuring detector/model degradation.
- If GALILEO doesn’t include multilingual stress-tests, Persuaficial suggests portability concerns across languages.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a “subtle manipulation” condition (rewrite-level minimal edits) and test whether our detectors/metrics degrade more than on overt manipulation.
- [ ] Consider an interpretable feature view (stylistic/linguistic) alongside LLM judges, to reduce evaluation circularity.
- [ ] Consider multilingual variants (at least EN + 1–2 others) for generalization.

## Quotes / details to potentially cite

- Persuaficial: “a high-quality multilingual benchmark covering six languages: English, German, Polish, Italian, French and Russian.”
- Key claim: “subtle LLM-generated persuasion consistently degrades automatic detection performance.”
- Dataset scale: “multilingual corpus of about 65,000 texts” with “4 models × 4 approaches = 16 generation configurations.”
- Feature analysis: “196 distinct linguistic features” using StyloMetrix.
