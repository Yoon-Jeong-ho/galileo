# Why Uncertainty Estimation Methods Fall Short in RAG: An Axiomatic Analysis

- Year: 2025
- Venue: arXiv (cs.IR)
- Authors: Heydar Soudani; Evangelos Kanoulas; Faegheh Hasibi
- URL: https://arxiv.org/abs/2505.07459
- BibTeX key (if we add it): soudani2025ue_rag_axioms
- Tags: rag, uncertainty-estimation, calibration, axioms, retrieval-noise

## One-sentence takeaway

Existing LLM uncertainty estimation methods behave inconsistently once retrieved context is injected (RAG), and an axiomatic set of “desired uncertainty behaviors” both explains the failures and yields a simple calibration wrapper that improves AUROC/correlation.

## What problem does it solve?

- In standard (no-RAG) prompting, many UE methods (white-box entropy/probability-based; black-box self-consistency, etc.) correlate with correctness.
- In RAG, the prompt includes non-parametric evidence that may be supportive, irrelevant, or misleading; it is unclear whether UE scores still mean anything.
- The paper argues UE-in-RAG should satisfy specific monotonicity/consistency properties with respect to whether retrieved context supports or contradicts the model’s answer.

## What is the core method / protocol?

- Evaluate a range of UE methods under multiple retrieval regimes:
  - “weak” synthetic retriever returning irrelevant docs
  - “ideal” retriever ranking the gold doc at top
  - standard retrievers with varying performance
- Propose an axiomatic framework: five constraints describing how uncertainty should change when moving from no-RAG (query only) to RAG (query + context), depending on whether context supports/contradicts the response and whether the response changes.
  - Example from intro/fig: if the response is correct both before and after RAG and the context supports it, uncertainty should *decrease* with RAG.
  - If the model changes from a wrong answer (no-RAG) to a supported correct answer (with RAG), uncertainty should also *decrease*.
- Show via experiments that no existing UE method satisfies all axioms; many violate several axioms frequently.
- Derive a lightweight “calibration function” (a multiplicative / coefficient-style wrapper around a base UE method) informed by the axioms, with multiple instantiations; improves stability and UE–correctness correlation.

## What are the key metrics?

- AUROC for separating correct vs incorrect answers using uncertainty scores (explicitly mentioned).
- “Correlation between uncertainty estimates and correctness” (as described in abstract/intro).
- Axiom satisfaction rate (how often a method’s uncertainty changes align with each axiom) as a diagnostic metric.

## What are the main results?

- UE performance “mainly deteriorates” when non-parametric knowledge is added (RAG prompts), even for UE methods that are improvements in no-RAG settings.
- Across datasets/retrievers, existing UE methods satisfy at most ~2 of the 5 axioms (per intro); violations explain suboptimal UE in RAG.
- The proposed axiom-guided calibration wrapper satisfies more axioms than baselines and improves AUROC / uncertainty–correctness association.

## How is this similar to GALILEO?

- Directly about *trust signals* (uncertainty/confidence) for LLM outputs when external evidence is present (RAG), which is a common setting for “reliability-aware” agent/pipeline designs.
- Provides a clean *evaluation lens* (axioms) rather than only reporting scalar UE metrics—useful for diagnosing failure modes in complex prompting setups.

## How is this different from GALILEO?

- Focuses on UE behavior and diagnostics in RAG; it is not primarily a benchmark of multi-turn interaction robustness or agentic behavior.
- Emphasizes axiomatic “should change” constraints rather than task-specific utility / decision-theoretic risk control.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO already includes multi-turn/agent settings, it likely covers richer interaction dynamics than single-shot RAG QA.
- If GALILEO evaluates downstream decision impact (abstention, escalation, etc.), that goes beyond AUROC-style UE evaluation.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently reports confidence/uncertainty only as a scalar metric, it may miss *behavioral inconsistency under evidence injection* (supporting vs contradicting retrieval) that this axiomatic view isolates.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add an “axiom checks” section for any GALILEO setting with external evidence (retrieval, tools, citations): define a small set of desired monotonicity constraints and report satisfaction rates.
- [ ] In analysis, break confidence failures into cases: (a) answer unchanged + context supports; (b) answer unchanged + context contradicts; (c) answer changes towards supported answer; (d) answer changes away from supported answer.
- [ ] Consider an axiom-inspired calibration layer (post-hoc scaling) for confidence scores when evidence is injected, and report whether it improves reliability without harming utility.

## Quotes / details to potentially cite

- Abstract: “Our framework introduces five constraints that an effective UE method should meet after incorporating retrieved documents into the LLM's prompt. Experimental results reveal that no existing UE method fully satisfies all the axioms…”
- Intro: “Most notably, improvements on the proposed UE methods in the literature do not add up when considering RAG setup.”
