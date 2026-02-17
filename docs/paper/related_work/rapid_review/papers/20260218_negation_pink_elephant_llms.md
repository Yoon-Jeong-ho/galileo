# Negation: A Pink Elephant in the Large Language Models' Room?

- Year: 2025
- Venue: arXiv (cs.CL)
- Authors: Tereza Vrabcová; Marek Kadlčík; Petr Sojka; Michal Štefánik; Michal Spiegel
- URL: https://arxiv.org/abs/2503.22395
- BibTeX key (if we add it): vrabcova2025negation
- Tags: negation, textual-entailment, NLI, multilingual, robustness, scaling

## One-sentence takeaway

Multilingual NLI benchmarks with paired negated hypotheses show LLM entailment accuracy is brittle to negation in a language- and context-dependent way, but larger models can be more robust than prior work suggested.

## What problem does it solve?

- Provides targeted datasets/analysis for “negation blindness” in LLM reasoning, especially beyond English.
- Separates *reasoning accuracy* from *robustness to negation* by using paired examples differing only by (verb) negation.

## What is the core method / protocol?

- Build two new multilingual textual entailment datasets:
  - **NoFEVER-ML**: based on FEVER-NLI style factual premises.
  - **NoSNLI-ML**: based on SNLI-style short image-caption premises.
- For each (premise, hypothesis), create a **paired negated hypothesis** (negating the main verb), such that if one is entailed, the other is contradicted (and vice versa).
- English negation generation: rule-based verb negation (spaCy-based) + **manual verification/fixes**.
- Translate the English datasets into **Czech, German, Ukrainian** (DeepL selected; random-sample validation described).
- Evaluate multiple LLMs (incl. different sizes within a family) on textual entailment; compare performance on non-negated vs negated hypotheses to quantify **accuracy drop / robustness gap**.

## What are the key metrics?

- Textual entailment accuracy (and the *difference* between accuracy on hypotheses with vs without negation).
- “Robustness to negation” operationalized as: minimal degradation when negation is introduced in the paired hypothesis.

## What are the main results?

- **Scaling can help**: contrary to some prior claims, larger models (within a family) can show improved ability/robustness for negation.
- **Language matters**: robustness and accuracy vary across languages; better results reported for more fixed word-order/projective languages (e.g., English) vs more flexible/non-projective ones (e.g., Czech/German).
- **Context matters**: premise length/explicitness affects robustness (longer/more explicit premises tend to help).

## How is this similar to GALILEO?

- Directly about **robustness of LLM reasoning under semantic operators** (negation) and how evaluation design can reveal brittleness.
- Emphasizes controlled contrast sets (paired examples) to isolate a single phenomenon—aligned with careful evaluation methodology.

## How is this different from GALILEO?

- Focuses on **textual entailment** datasets/benchmarks and multilingual effects, not on GALILEO’s specific task setting.
- Primarily diagnostic/benchmarking rather than proposing a new training-time method to improve reasoning.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO includes mechanistic or algorithmic components beyond dataset probing, it can claim a more *interventionist* contribution than this primarily evaluative paper.

## Where GALILEO is weaker / needs to improve

- If GALILEO evaluation is mostly English-only or lacks contrast sets around negation, this paper highlights a gap: **multilingual** + **minimal-pair negation** stress tests.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a small **negation minimal-pair** slice to GALILEO eval (premise fixed; hypothesis negated), report both raw accuracy and “robustness gap”.
- [ ] If we claim robustness, consider reporting results across at least 2 languages (even if via translation) or explicitly scope claims to English.
- [ ] In related work, cite as evidence that (a) negation remains hard, (b) scaling effects are nuanced, (c) language structure/context length affect robustness.

## Quotes / details to potentially cite

- Paper framing: negation is a “pink elephant in the room” for LLM reliability/logical reasoning.
- Contribution claim: release of **NoFEVER-ML** and **NoSNLI-ML** (English/Czech/German/Ukrainian) with paired examples differing in negation.
- Finding claim: larger model sizes can improve negation handling; robustness depends on language and on premise length/explicitness.
