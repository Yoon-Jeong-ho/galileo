# Visual reasoning consistency and robustness analysis of multimodal LLMs

- Year: 2025
- Venue: Pattern Recognition (Elsevier)
- Authors: (not captured in abstract snippet; fill from PDF later)
- URL: https://www.sciencedirect.com/science/article/pii/S0031320325014281
- BibTeX key (if we add it): visual_reasoning_consistency_2025
- Tags: multimodal, robustness, consistency, evaluation, positional-bias, abstention, uncertainty

## One-sentence takeaway

Introduces an open-access benchmark for **multi-image visual reasoning** that measures **consistency under answer-order permutations**, **positional bias**, and **uncertainty via rejection/abstention**, proposing **entropy** as a compact stability metric.

## What problem does it solve?

- Prior multimodal LLM evals often focus on single-image accuracy and miss:
  - contextual / multi-image reasoning
  - reasoning stability (sensitivity to answer order / variant reorderings)
  - uncertainty calibration (when to abstain)

## What is the core method / protocol?

- Benchmark combining:
  - **multi-image reasoning tasks** (8 tasks)
  - **rejection-based evaluation** (models can abstain)
  - **positional bias detection** (sensitivity to answer ordering)
  - consistency analysis across **reordered answer variants**
- Evaluates 8 models (examples mentioned in abstract: Grok 3, ChatGPT-4o, Gemini 2.0 Flash Experimental, DeepSeek Janus).

## What are the key metrics?

- **Overall accuracy**
- **Rejection accuracy** (accuracy when the model chooses to answer vs reject; or correctness of rejection decisions—terminology to confirm from full text)
- **Entropy** over predictions across reordered answer variants (higher entropy = less consistent / more order-sensitive)
- **Abstention rate** (tendency to avoid uncertain answers)
- **Positional bias** indicators (captured via the above permutation/entropy + rejection behavior)

## What are the main results?

(From the abstract)

- ChatGPT-o1: **82.5%** overall accuracy, **70.0%** rejection accuracy.
- Gemini 2.0 Flash Experimental: **70.8%** overall accuracy.
- Janus models show poorer bias/uncertainty behavior:
  - rejection accuracy: Janus 7B **25.0%**, Janus 1B **30.0%**
  - high entropy: Janus 7B **0.839**, Janus 1B **0.787**

## How is this similar to GALILEO?

- Same high-level motivation: robustness is not just “final accuracy”, but **stability under perturbations**.
- Treats evaluation as multi-context (multi-image) rather than single input.
- Explicitly calls out **bias from evaluation protocol** (positional bias), analogous to our concern that multi-turn protocols can create artifacts (order/primacy effects).

## How is this different from GALILEO?

- Focus is **multimodal / visual reasoning**, not multi-turn persuasion/social pressure.
- “Stability” is measured via **answer-option/variant reordering** rather than **time-to-failure across dialogue turns**.
- Uses **rejection/abstention** as a first-class uncertainty behavior; GALILEO is more about drift/flip dynamics under pressure.

## Where GALILEO is stronger / cleaner (if true)

- Provides a more direct model of **interactive, sequential** failure (turn-level drift / flip / recovery), which this work does not emphasize in the abstract.
- Our controls can more cleanly separate **pressure-driven drift** from **evidence-driven correction** (depending on final GALILEO protocol).

## Where GALILEO is weaker / needs to improve

- We should be careful to report **uncertainty / abstention** behaviors, or at least show that any “robustness” gains are not coming from trivial refusal.
- We likely need better coverage of **protocol-induced bias** (order effects) as a general evaluation pitfall.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add an “order-sensitivity” check: randomize ordering of pressure operators / rebuttal styles and report robustness variance.
- [ ] Consider an **entropy-of-trajectory** or **entropy-over-variants** metric as a compact stability measure (conceptual analogue of their permutation entropy).
- [ ] Add explicit **abstention / refusal calibration** reporting (in the spirit of rejection accuracy), to avoid “safe by refusing” or “robust by hedging” confounds.

## Quotes / details to potentially cite

- “Traditional evaluations of multimodal large language models (LLMs) have been limited by their focus on single-image reasoning, failing to assess crucial aspects like contextual understanding, reasoning stability, and uncertainty calibration.”
- “We further introduce entropy as a novel metric for quantifying reasoning consistency across reordered answer variants and abstention rate to measure a model’s tendency to avoid uncertain answers.”
