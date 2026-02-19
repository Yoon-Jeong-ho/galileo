# When Models Examine Themselves: Vocabulary-Activation Correspondence in Self-Referential Processing

- Year: 2026
- Venue: arXiv
- Authors: Zachary Dadfar (per arXiv listing)
- URL: https://arxiv.org/abs/2602.11358
- BibTeX key (if we add it): dadfar2026models
- Tags: self-reference, introspection, interpretability, activation-steering

## One-sentence takeaway

Under a long-format “self-examination” prompt, models invent introspective vocabulary whose *usage* correlates with specific activation-dynamics metrics, and the authors extract a layer-localized “self-referential vs descriptive” activation direction that can steer that vocabulary.

## What problem does it solve?

- Clarifies whether LLM “introspective” language is purely confabulation vs. is (sometimes) tethered to measurable internal computation.
- Provides a protocol to reliably elicit extended self-referential behavior and a mechanistic handle (an activation-space direction) to study it.

## What is the core method / protocol?

- **Pull Methodology**: format-engineered prompt that induces **1,000 sequential self-observations (“pulls”)** in a single inference pass; model invents vocabulary for what it “finds” (no target words in prompt).
- **Activation direction extraction**: construct a vector that separates **self-referential** vs **descriptive** processing for the *same token* in different contexts (example token given: “glint”); show transfer to new introspective content.
- **Causal steering**: add the direction into activations at a specific depth to amplify introspective mode / vocabulary.
- **Specificity controls**: check that the *same vocabulary* in non-self-referential contexts does **not** show the same activation correspondence, despite being much more frequent.
- Cross-model check: replicate the vocabulary↔activation-metric mapping phenomenon in **Qwen 2.5-32B** with different vocab/metrics.

## What are the key metrics?

- Correlation between introspective-vocabulary frequency and activation-dynamics summaries, notably:
  - **Lag-1 activation autocorrelation** (“loop” vocabulary).
  - **Activation variability** (“shimmer” vocabulary under steering).
  - (In Qwen) other activation-dynamics metrics (paper mentions spectral power for some vocab).
- Effect sizes / separations reported as d-values for direction transfer/steering (as summarized in intro).
- Cosine similarities to show direction properties (incl. orthogonality vs refusal direction).

## What are the main results?

- The extracted **self-referential direction** is:
  - **Layer-localized** at ~**6.25% of depth** (reported for Llama 8B and 70B).
  - **Orthogonal to refusal** direction (cos sim reported ~0.063).
  - **Causally effective** for steering introspective output.
  - Claimed to preserve refusal behavior under steering.
- **Vocabulary-activation correspondence** (self-referential mode only):
  - “loop” vocabulary correlates with higher activation autocorrelation (reported r≈0.44, p=0.002).
  - Under steering, “shimmer” vocabulary correlates with increased activation variability (reported r≈0.36, p=0.002).
  - In descriptive controls, the same words show near-zero correlation despite higher frequency.
- Another architecture (Qwen 2.5-32B) shows analogous correspondence but with **different invented vocab** and **different activation metrics**, suggesting the effect is about processing mode rather than specific tokens.

## How is this similar to GALILEO?

- Mechanistic/causal flavor: uses **directions in representation space** + **steering** to argue some behavioral phenomenon is grounded in activations.
- Emphasizes the importance of **controls** (same surface words, different context) to demonstrate specificity—similar spirit to distinguishing genuine updates vs surface-level compliance.

## How is this different from GALILEO?

- Focuses on **self-referential introspection** rather than user-pressure / multi-turn belief drift (GALILEO’s core).
- Relies on **internal access** (activations, layer interventions) as primary evidence; less directly applicable to black-box-only settings.
- The task/prompt is highly stylized (1,000-step self-examination) and may be far from practical dialogue conditions.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO is positioned as behaviorally grounded / protocol-driven under realistic pressure: likely stronger on **ecological multi-turn interactions** and external validity.
- If GALILEO aims for black-box methods: stronger on **no-internals-required** evaluation/mitigation (this paper is internal-access heavy).

## Where GALILEO is weaker / needs to improve

- If GALILEO discusses “self-report” or “introspection” at all, this paper raises the bar: we may need tighter language about when self-report is informative vs confabulated.
- Potential missed angle: **mode-dependent gating** (self-referential vs descriptive) could inspire a similar “mode switch” hypothesis for pressure compliance vs evidence-based revision.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a brief related-work paragraph: *self-report can correlate with internal state under specific elicitation protocols*; cite as a counterexample to blanket “LLM introspection is useless.”
- [ ] Consider a conceptual analogy in discussion: pressure episodes might induce a distinct “interaction mode” (social-compliance mode) that could be separable/decodable (even if we do not use internals).
- [ ] If we ever add an interpretability appendix: replicate a lightweight version—decode a “pressure vs evidence” direction and test orthogonality vs refusal/sycophancy directions.

## Quotes / details to potentially cite

- Pull Methodology described as eliciting **1,000 sequential observations** of model processing within a **single inference pass**, producing **3,000–30,000 tokens** of sustained self-referential content per run.
- Direction properties (intro summary): orthogonal to refusal (cos sim ~0.063), localized at ~6.25% depth, and vocabulary↔activation correlations like “loop”↔autocorrelation (r≈0.44) and “shimmer”↔variability (r≈0.36) in self-referential mode but not descriptive controls.
