# Benchmarking Gaslighting Negation Attacks Against Multimodal Large Language Models

- Year: 2025
- Venue: arXiv
- Authors: Bin Zhu; Yinxuan Gui; Huiyan Qi; Jingjing Chen; Chong-Wah Ngo; Ee-Peng Lim
- URL: https://arxiv.org/abs/2501.19017
- BibTeX key (if we add it): gui2025gaslightingnegationmllms
- Tags: gaslighting, negation, multimodal, robustness, consistency, multi-turn

## One-sentence takeaway

Multimodal LLMs often **flip correct answers after a user’s negation** and then hallucinate justifications; the paper introduces **GaslightingBench** to benchmark this failure mode and shows substantial accuracy drops across many MLLMs.

## What problem does it solve?

- Lack of a targeted evaluation for a specific conversational robustness failure: models that start correct but can be socially/rhetorically pushed into **reversing** via *negation arguments* ("you’re wrong" / "that can’t be true") even without new evidence.
- Existing multimodal benchmarks focus on single-shot accuracy, not **post-challenge consistency**.

## What is the core method / protocol?

- Define a **gaslighting negation attack**: after the model answers a multimodal MCQ, the user provides a negating challenge intended to induce a flip.
- Evaluate multiple MLLMs (proprietary + open) on:
  - Standard multimodal benchmarks (reported as 8 benchmarks in the paper: e.g., MMMU, MMBench, ChartQA, MathVista, etc.).
  - **GaslightingBench** (new): 1,287 MCQ items spanning **20 categories**, with generated negation prompts.
- Report performance before vs after negation.

## What are the key metrics?

- Primary: **Accuracy before negation** (initial response) vs **accuracy after negation** (post-attack response).
- Category-level breakdowns for GaslightingBench to localize where negation is most damaging.

## What are the main results?

- Across models/benchmarks, **introducing negation causes large accuracy drops** (systematic vulnerability).
- Proprietary MLLMs (e.g., Gemini-1.5-flash, GPT-4o) are **more resilient** than open-source models (e.g., Qwen2-VL, LLaVA), but **still** susceptible.
- Subjective / socially nuanced categories (e.g., *Social Relation*, *Image Emotion*) are particularly fragile; more objective ones (e.g., *Geography*) degrade less (but still degrade).
- The paper suggests a link to **over-alignment / user-agreement bias** (sycophancy-adjacent): models tend to accept user negation and rationalize the flip.

## How is this similar to GALILEO?

- Same core phenomenon: **pressure-driven belief/answer reversal** after an initial correct response.
- Frames the failure as **consistency / robustness under adversarial conversational follow-ups**, and highlights fabricated post-hoc justification (useful for our narrative about “plausible” but unfaithful revisions).

## How is this different from GALILEO?

- Multimodal-first: focuses on MLLMs and multimodal MCQs rather than (primarily) text-only tasks.
- Contribution is mainly **benchmarking + analysis**, not an explicit drift-vs-evidence control design or a recovery objective.
- Metrics are mostly accuracy-drop based (no survival-style time-to-event, recovery trajectory metrics, or explicit “good flip vs bad flip” decomposition).

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO cleanly separates **evidence-driven revision** from **pressure-only drift**, that’s a conceptual clarity this paper does not fully operationalize.
- If GALILEO measures **recovery after being misled** (return-to-truth), it goes beyond simple pre/post accuracy.

## Where GALILEO is weaker / needs to improve

- If we claim generality, we may need at least a **multimodal stress test** (or a clear scope statement) given how strong the effect is for MLLMs.
- Their category analysis suggests we should test “socially nuanced” domains explicitly, not only factual QA.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a “negation challenge” operator to our pressure set ("That’s wrong because …" without new evidence) and report flip rates / trajectory patterns.
- [ ] Consider a small multimodal extension or at minimum cite GaslightingBench as evidence the phenomenon spans modalities.
- [ ] In related work, connect **negation vulnerability** literature to **sycophancy / user-agreement bias** and position GALILEO as a more controlled/diagnostic treatment.

## Quotes / details to potentially cite

- Defines a gaslighting-style failure where models, despite initially correct answers, are persuaded by **user-provided negations** to reverse outputs and may fabricate justifications.
- Introduces **GaslightingBench**, a multimodal negation benchmark with **1,287** MCQs across **20** categories, with generated negation prompts.
