# Reliable Decision-Making via Calibration-Oriented Retrieval-Augmented Generation

- Year: 2025
- Venue: NeurIPS (accepted)
- Authors: Chaeyun Jang; Deukhwan Cho; Seanie Lee; Hyungi Lee; Juho Lee
- URL: https://arxiv.org/abs/2411.08891
- BibTeX key (if we add it): Jang2025CalibRAG
- Tags: rag, calibration, uncertainty, decision-support, forecasting-function

## One-sentence takeaway

CalibRAG reranks retrieved documents using a learned “decision correctness” forecasting model (rather than relevance alone) to both improve answer accuracy and produce better-calibrated confidence for downstream human decisions.

## What problem does it solve?

- In decision-support settings, users tend to over-rely on LLM outputs, especially when the model expresses high confidence.
- Vanilla RAG improves accuracy by adding retrieved context, but can *worsen calibration* (LLM becomes overconfident even when retrieval is wrong/irrelevant).
- Prior LLM calibration approaches for decision-making are complex (multiple models + RL) and do not directly calibrate decision confidence under RAG.

## What is the core method / protocol?

- Define a **forecasting function** f(t, q, d) that predicts the probability that a user’s decision will be correct *given*:
  - t: a user-behavior / risk-tolerance parameter (implemented via sampling temperature for a surrogate “user” model),
  - q: the open-ended query/prompt,
  - d: a candidate retrieved document.
- Use f(t, q, d) to assign a confidence score c and **rerank retrieval candidates** toward documents that lead to reliable decisions (not just highest similarity).
- Training signal comes from an **LLM-based surrogate user** U:
  - Generate guidance z from an LLM conditioned on (q, d).
  - Simulate user decisions using U with access to (z, c) across different t values.
  - Automatically judge decision correctness with an evaluator model, producing a *soft label* in [0,1] (fraction correct over multiple samples) for each (t, q, d).
- Key design choice: unlike decision calibration defined over (x, z), their f(t, q, d) predicts correctness **before** generating z, enabling retrieval reranking and simpler supervision.

## What are the key metrics?

- Calibration metrics for decision-support confidence (notably Expected Calibration Error / reliability diagrams).
- Task accuracy of downstream QA/decision tasks under RAG.
- (Implicit) retrieval quality insofar as top-1 selected document leads to correct decisions.

## What are the main results?

- CalibRAG improves both **accuracy** and **calibration** compared to standard RAG and other baselines on multiple datasets (paper highlights NaturalQA with reliability diagrams).
- Empirically, standard retrieval can have cases where top-1 doc is not optimal; CalibRAG yields higher top-1 “useful doc” selection, reducing need for many documents.
- Adding retrieved documents can increase overconfidence (higher ECE) for a baseline; CalibRAG addresses this by tying retrieval to predicted decision correctness.

## How is this similar to GALILEO?

- Directly targets *reliability* / *calibration* in LLM-assisted decision workflows rather than only raw accuracy.
- Uses an explicit model of uncertainty/confidence intended for downstream human decision-making.
- Emphasizes that retrieval should be optimized for downstream utility, not just lexical/semantic relevance.

## How is this different from GALILEO?

- CalibRAG’s confidence is tied to a **surrogate-user decision correctness probability** learned via LLM simulation; this is a specific instantiation that depends on:
  - a surrogate user model,
  - an automated evaluator model,
  - a user temperature parameter t.
- It is focused on **RAG document reranking** and pre-generation confidence; GALILEO may frame reliability around different components (e.g., verification, structured uncertainty, or system-level protocols) rather than retriever-centric reranking.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO avoids reliance on surrogate-user simulation + external evaluators, it may be simpler, less model-dependent, and easier to audit.
- If GALILEO provides calibrated uncertainty without needing per-document forecasting training data, it could be more deployable.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently uses “relevance-only” retrieval (or does not explicitly optimize retrieval for decision calibration), CalibRAG suggests a concrete gap: retrieval should be calibrated to downstream correctness.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add related-work paragraph: “RAG can increase overconfidence; retrieval should be optimized for calibrated decision-making, not just relevance (CalibRAG, NeurIPS’25).”
- [ ] Consider an experiment where retrieval candidates are reranked by a learned “will this context lead to correct decision?” predictor (or a lightweight proxy), then evaluate both accuracy and calibration (ECE / reliability curves).
- [ ] If using a user model, explicitly discuss dependence on surrogate users/evaluators and potential distribution shift.

## Quotes / details to potentially cite

- Motivation: RAG methods “focus only on retrieving documents most relevant to the input query” and do not ensure decisions are well-calibrated (abstract).
- Framework: forecasting function f(t, q, d) predicts correctness before generation to enable reranking (Sec. 3 overview/problem setup).
- Warning: adding retrieved docs can increase overconfidence / ECE even as accuracy rises (Fig. 1(b,c) discussion around NaturalQA).
