# Large Language Models Meet Stance Detection: A Survey of Tasks, Methods, Applications, Challenges and Future Directions

- Year: 2025 (arXiv v1; updated v2 Jan 2026)
- Venue: arXiv (cs.CL)
- Authors: Nagendra Kumar et al.
- URL: https://arxiv.org/abs/2505.08464
- BibTeX key (if we add it): kumar2025llmstance (suggested)
- Tags: stance-detection, survey, llms, evaluation, robustness-adjacent

## One-sentence takeaway

A broad survey that organizes LLM-based stance detection by learning setup, modality, and target-relationship, and highlights evaluation practices and open challenges like implicit stance and bias.

## What problem does it solve?

- Consolidates (and updates) the fragmented stance-detection literature specifically in the era of LLMs.
- Provides a taxonomy to compare methods across: learning regime (supervised/unsupervised/few-shot/zero-shot), modality (uni/multi/hybrid), and target-relationship (in-target/cross-target/multi-target).
- Summarizes datasets, evaluation techniques, and application areas (misinformation, political analysis, public health, moderation).

## What is the core method / protocol?

- Survey / systematization paper (not a new model).
- Proposes a taxonomy for “LLM-based stance detection approaches” along 3 dimensions:
  - learning methods (supervised, unsupervised, few-shot, zero-shot)
  - data modalities (unimodal, multimodal, hybrid)
  - target relationships (in-target, cross-target, multi-target)
- Reviews evaluation techniques + benchmark datasets, then discusses challenges and future directions.

## What are the key metrics?

- Not specified in the abstract; stance detection commonly uses accuracy / macro-F1 (and sometimes class-wise F1), plus robustness/generalization evaluations (e.g., cross-target / cross-domain).
- This survey is more useful for “what metrics are used” than for introducing a novel metric.

## What are the main results?

- Main “result” is the organized mapping of the space (taxonomy) and a synthesis of trends/limitations.
- Emphasizes challenges:
  - implicit stance expression
  - cultural biases
  - computational constraints
- Points to future directions:
  - explainable stance reasoning
  - low-resource adaptation
  - real-time deployment frameworks

## How is this similar to GALILEO?

- Overlaps conceptually on *stance/belief as a latent variable* and how models express (or hide) it under different contexts.
- The “target-relationship” lens (in-target vs cross-target vs multi-target) is adjacent to GALILEO’s concern with *generalization of behavior across contexts/pressures*.
- Highlights evaluation pitfalls and bias issues that resemble GALILEO’s “evaluator brittleness / artifact” concerns.

## How is this different from GALILEO?

- Focuses on stance detection as a downstream NLP task (classification/recognition), rather than *multi-turn robustness under pressure* or “drift vs evidence-driven revision”.
- Primarily single-instance or short-context task framing (typical stance datasets), not multi-round interactive protocols where stance can evolve due to persuasion.
- Does not (from abstract) propose survival/time-to-failure style instability metrics, intervention/recovery protocols, or controlled pressure conditions.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO provides interactive, multi-turn protocols with explicit controls (pressure vs evidence), it is more directly targeted to measuring *instability and recovery* than stance-detection benchmarks.
- GALILEO can likely offer clearer causal structure: what changed the model’s stance (new evidence vs social pressure) and when.

## Where GALILEO is weaker / needs to improve

- GALILEO may under-cover the breadth of stance-detection datasets, application domains, and cross-target evaluation setups summarized here.
- If GALILEO lacks multimodal/hybrid settings, this survey suggests modalities as an axis worth addressing or explicitly scoping out.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider framing some GALILEO analyses using “target relationships”: define what “target” is in GALILEO (topic, claim, persona, authority source) and test in-target vs cross-target pressure generalization.
- [ ] Add a short related-work paragraph that positions GALILEO vs stance detection: stance detection (recognition) vs stance stability/revision under interaction (behavioral robustness).
- [ ] Scan the survey’s dataset section later for candidates that could be adapted into GALILEO-style multi-turn pressure protocols (especially implicit stance, culturally-sensitive topics).

## Quotes / details to potentially cite

- “We present a novel taxonomy for LLM-based stance detection approaches, structured along three key dimensions: (1) learning methods … (2) data modalities … and (3) target relationships …”
- Challenges called out: “implicit stance expression, cultural biases, and computational constraints” (plus future directions: “explainable stance reasoning, low-resource adaptation, and real-time deployment frameworks”).
