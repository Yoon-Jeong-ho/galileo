# XIFBench: Evaluating Large Language Models on Multilingual Instruction Following

- Year: 2025
- Venue: NeurIPS 2025 (Datasets and Benchmarks Track)
- Authors: Zhenyu Li; Kehai Chen; Yunfei Long; Xuefeng Bai; Yaoyin Zhang; Xuchen Wei; Juntao Li; Min Zhang
- URL: https://arxiv.org/abs/2503.07539
- BibTeX key (if we add it): XIFBench2025
- Tags: instruction-following, multilingual, benchmark, constraints, llm-judge

## One-sentence takeaway

XIFBench is a 6-language, constraint-rich instruction-following benchmark that uses English “requirements” as semantic anchors to make multilingual constraint-following evaluation more reliable and comparable.

## What problem does it solve?

- Existing multilingual instruction-following evals are often (a) translated from English-only benchmarks and (b) scored coarsely, making it hard to attribute failures to specific constraint types or to compare fairly across languages.
- Translating evaluation criteria/requirements can introduce semantic drift that confounds cross-lingual comparisons.

## What is the core method / protocol?

- Dataset: 558 instructions with 0–5 additional constraints, with a taxonomy of 5 categories / 21 dimensions:
  - Content, Style, Situation, Format, Numerical.
- Languages: English + {Chinese, Russian, Arabic, Hindi, Swahili} spanning high/medium/low resource.
- Three key “reliability” additions:
  1) **Cultural accessibility** annotation for instructions (culturally universal vs specific).
  2) **Constraint-level translation validation** (requirement-wise semantic preservation checks via back-translation + LLM).
  3) **Requirement-based evaluation with English requirements as anchors**: keep the evaluation requirements in English for all languages rather than translating them.
- Evaluation: LLM-as-judge produces observations + binary YES/NO per requirement (InfoBench-style).

## What are the key metrics?

- **RFR (Requirement Following Rate):** fraction of individual requirements satisfied.
- **IFR (Instruction Following Rate):** fraction of instructions where *all* requirements are satisfied.
- Also reports evaluator–human agreement rates under different evaluation setups (direct scoring vs translated requirements vs anchored English requirements).

## What are the main results?

- Clear **resource-level performance gradient**: high-resource languages are more stable; low-resource (Hindi/Swahili) show steep IFR drops (sometimes near 0), even when RFR remains moderately high.
- Large **RFR–IFR gap**, especially in low-resource languages: models can satisfy many individual constraints but fail to satisfy *all* constraints simultaneously.
- Using **English-anchored requirements** improves evaluator reliability/consistency vs translating requirements:
  - Agreement study reports higher avg agreement and lower cross-lingual variance for “Eng. Reqs.” vs “Trans. Reqs.”.
- Constraint-category analysis suggests **style/situation** constraints degrade most as language resources/model capability drop, while **format/numerical** are more robust across languages.

## How is this similar to GALILEO?

- Shares the theme of **fine-grained protocol/metrics** for behavioral evaluation (breaking a global success criterion into atomic checks).
- Highlights the importance of **stable evaluation anchors** to avoid confounds (analogous to GALILEO’s need for robust scoring under distribution shifts / multi-turn artifacts).

## How is this different from GALILEO?

- XIFBench is primarily **single-turn instruction-following** (with varying constraint counts) rather than GALILEO’s focus on **multi-turn instability/robustness under pressure**.
- Target phenomenon is **multilingual constraint adherence**, not “belief revision vs drift,” “sycophancy/persuasion,” or turn-by-turn survival dynamics.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO uses survival/time-to-failure or turn-level dynamics: that is a cleaner fit for *multi-turn robustness*, which XIFBench does not directly address.

## Where GALILEO is weaker / needs to improve

- Multilinguality: if GALILEO claims general robustness but evaluates mostly English, XIFBench shows why cross-lingual comparisons can be brittle without explicit anchoring/validation.
- Evaluation anchors: XIFBench’s “English requirements as anchors” is a concrete technique to reduce translation confounds.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding a **“requirements anchor”** concept to GALILEO’s evaluation writeup: keep scoring criteria fixed in a single language / canonical representation, and treat translations as *stimuli* only.
- [ ] If we ever expand GALILEO beyond English, adopt a **constraint-/requirement-level translation validation** step (spot-check + back-translation + human/LLM review).
- [ ] Use the RFR vs IFR framing as a writing device: analog for GALILEO could be “per-turn requirement satisfaction” vs “episode success” to make failure modes interpretable.

## Quotes / details to potentially cite

- “XIFBench … 558 instructions with 0–5 additional constraints … five categories (Content, Style, Situation, Format, and Numerical) … six languages …” (abstract).
- “We retain the original English evaluation requirements across all languages, using them as fixed semantic anchors … ensures cross-linguistic comparability.” (evaluation protocol section).
- Metrics definition: Requirement Following Rate (RFR) vs Instruction Following Rate (IFR).