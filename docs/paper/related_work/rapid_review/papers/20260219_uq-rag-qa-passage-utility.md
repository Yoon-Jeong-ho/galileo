# Uncertainty Quantification in Retrieval Augmented Question Answering

- Year: 2025
- Venue: TMLR (09/2025)
- Authors: Laura Perez-Beltrachini; Mirella Lapata
- URL: https://arxiv.org/abs/2502.18108
- BibTeX key (if we add it): perez-beltrachini2025_uq_rag_qa
- Tags: rag, uncertainty, qa, passage-utility, calibration

## One-sentence takeaway

Train a small passage-utility predictor (based on the target QA model’s observed accuracy+entailment behavior) and aggregate predicted utilities to estimate RAG QA answer uncertainty more efficiently than sampling-based methods.

## What problem does it solve?

- In retrieval-augmented QA, uncertainty estimation is tricky because errors can come from bad/irrelevant/misleading retrieved passages, not just the QA model’s internal uncertainty.
- Existing uncertainty methods (e.g., output-sampling/entropy, self-reported confidence) can be expensive at inference time and may show reduced variation / overconfidence in RAG settings.
- The paper targets: predicting when a RAG QA model is likely to be wrong, by assessing whether the provided retrieved context is actually *useful* for answering.

## What is the core method / protocol?

- Define RAG QA: model M generates answer y given question x and retrieved set R of passages.
- Key hypothesis: a passage’s *utility* for the *target QA model* is indicative of the model’s chance of answering correctly when prompted with retrieved evidence.
- Construct training labels for each (x, p) passage using the target QA model’s behavior:
  - Utility v_M is computed from:
    - Accuracy a(y) in {0,1} (is generated answer correct?)
    - Entailment e(y) in [0,1] (does passage support the generated answer?), from an NLI model
  - Combine as: v_M = (a(y) + e(y)) / 2
- Train a lightweight utility predictor with a *pairwise ranking/contrastive* objective across passages retrieved for the same question:
  - Encourage higher predicted utility for more-useful passages with a margin-ranking loss.
  - Model: Siamese network with a BERT encoder + pooling + MLP layers to output a scalar utility score per (x,p).
- Use predicted passage utilities over the set R to derive an uncertainty estimate for the QA model when answering with R (details beyond the excerpt, but the core is “aggregate predicted passage utilities → uncertainty”).

## What are the key metrics?

- Utility label components:
  - Answer correctness/accuracy (dataset evaluator)
  - Passage-to-answer entailment score (NLI)
- Downstream uncertainty quality:
  - How well the uncertainty estimate predicts answer correctness / detects incorrect answers (paper compares vs sampling-based methods and simpler information-theoretic baselines).
- Practical metric emphasized:
  - Test-time efficiency / latency vs sampling multiple answers.

## What are the main results?

- Across 6 short-form information-seeking QA datasets, the utility-based uncertainty estimator is comparable to or outperforms sampling-based uncertainty methods, while being more efficient at test time.
- Empirical observation: in RAG QA, sampling-based methods can exhibit less answer variation (even when wrong), reducing their advantage; models can be confidently wrong when retrieved passages misleadingly “support” an incorrect answer.

## How is this similar to GALILEO?

- Treats retrieval context quality (usefulness/sufficiency) as a first-class driver of downstream behavior.
- Uses an auxiliary predictor trained from observed target-model behavior (direct error/utility prediction) rather than relying purely on model logprobs or sampling.
- Frames uncertainty as a function of (question, retrieved evidence), not just the question.

## How is this different from GALILEO?

- Focus is specifically uncertainty estimation for short-form RAG QA; not primarily about improving retrieval, reasoning, or long-form generation.
- Utility labels depend on a *target QA model* plus external evaluators (answer correctness) and optionally an NLI model; this is closer to supervised/observational calibration than “intrinsic” uncertainty.
- Uses a BERT-based Siamese scorer for (x,p) utility; GALILEO may use different encoders/objectives or operate at different granularities (e.g., multi-passage reasoning, plan/trace, etc.).

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO already has a principled notion of evidence attribution or robustness to misleading context, it may provide cleaner causal diagnostics than “observed accuracy+NLI entailment” labels.
- If GALILEO operates without requiring an extra NLI model, it may be simpler to deploy.

## Where GALILEO is weaker / needs to improve

- If GALILEO lacks a lightweight “context utility” predictor, it may be missing an inexpensive signal for gating / abstention.
- If GALILEO’s uncertainty relies on sampling or heavy LLM-based judging, this work suggests a cheaper learned proxy can be competitive.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding a “passage utility” head/model trained from observed outcomes (correct vs incorrect) for the target system; use it as an uncertainty signal and/or to re-rank/filter retrieved passages.
- [ ] Evaluate whether sampling-based uncertainty signals degrade in our retrieval setting (reduced answer variation, confident wrong answers supported by misleading context).
- [ ] In related work: cite this as “uncertainty via passage utility prediction” and contrast with sampling/entropy and self-reported confidence.

## Quotes / details to potentially cite

- Key idea: “quantify QA model uncertainty via estimating the utility of the passages it is provided with.”
- Utility definition: v_M = (a(y) + e(y)) / 2, where accuracy is correctness of generated answer and entailment measures whether passage supports the answer.
- Training: pairwise margin-ranking loss over passages retrieved for the same question; BERT-based Siamese encoder + MLP to score (x,p) utility.
