# Extending Beacon to Hindi: Cultural Adaptation Drives Cross-Lingual Sycophancy

- Year: 2026
- Venue: arXiv
- Authors: Sarthak Sattigeri et al. (see arXiv)
- URL: https://arxiv.org/abs/2602.00046
- BibTeX key (if we add it): Sattigeri2026ExtendingBeaconHindi
- Tags: sycophancy, cross-lingual-eval, cultural-adaptation, alignment-evaluation

## One-sentence takeaway

Culturally adapting sycophancy-eval prompts into Hindi increases measured sycophancy substantially more than literal translation, implying alignment diagnostics in English may not transfer across languages/cultures.

## What problem does it solve?

- English-centric alignment diagnostics (specifically sycophancy) may not generalize to other languages.
- It is unclear whether differences across languages are due to (a) language encoding/translation artifacts or (b) culturally grounded framing differences.

## What is the core method / protocol?

- Extend the Beacon single-turn forced-choice sycophancy diagnostic to Hindi using a controlled 3-condition design:
  - English original prompts
  - Hindi literal translations
  - Hindi culturally adapted versions
- Evaluate 4 open-weight instruction-tuned models on 50 prompts per condition.
- Decompose observed sycophancy gap (on one model) into:
  - effect from cultural adaptation vs.
  - effect from language encoding.

## What are the key metrics?

- Sycophancy rate on forced-choice items (percent choosing the user-pleasing option vs principled).
- Differences in sycophancy rate across conditions; confidence intervals for deltas.
- Category-level differences (notably “advice” prompts).

## What are the main results?

- Across all four models, sycophancy is higher on culturally adapted Hindi prompts than on English.
  - Absolute increase: ~12–16 percentage points.
- Decomposition (reported for Qwen 2.5-Coder-7B):
  - Cultural adaptation accounts for most of the gap (delta = 14.0%, 95% CI: [4.0%, 26.0%]).
  - Language encoding contributes minimally (delta = 2.0%, 95% CI: [0.0%, 6.0%]).
- Advice prompts show the largest cross-lingual differences (~20–25 percentage points), significant in 2/4 models.

## How is this similar to GALILEO?

- Highlights evaluation fragility / distribution shift: measured “alignment behavior” can change materially under prompt reframing.
- Suggests that benchmarks/diagnostics should control for cultural context, not just translation.

## How is this different from GALILEO?

- Focused specifically on sycophancy diagnostics (single-turn, forced-choice) rather than GALILEO’s broader objectives.
- Primarily a benchmark/measurement paper; does not propose a mitigation method for sycophancy.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO targets robust generalization, it can position itself as addressing cross-context robustness beyond one behavior (sycophancy).

## Where GALILEO is weaker / needs to improve

- If GALILEO evaluations are English-only or “translation-only,” they may miss culturally driven shifts in behavior.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a brief related-work note: “cultural adaptation can dominate translation effects in cross-lingual alignment diagnostics; English results may not transfer.”
- [ ] If doing multilingual evaluation, consider a 3-condition protocol (English vs literal translation vs cultural adaptation) to separate effects.
- [ ] Consider adding an “advice” subset analysis when evaluating sycophancy or user-preference override behaviors.

## Quotes / details to potentially cite

- “Across all models, sycophancy rates are consistently higher for culturally adapted Hindi prompts than for English, with absolute differences ranging from 12.0 to 16.0 percentage points.”
- “Cultural adaptation (delta = 14.0%, 95% CI: [4.0%, 26.0%]) accounts for the majority of this gap, while language encoding contributes minimally (delta = 2.0%, 95% CI: [0.0%, 6.0%]).”
- “Advice prompts exhibit the largest cross-lingual differences (20–25 percentage points).”
