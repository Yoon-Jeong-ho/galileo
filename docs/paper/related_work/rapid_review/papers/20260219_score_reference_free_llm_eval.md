# Score: Specificity, Context Utilization, Robustness, and Relevance for Reference-Free LLM Evaluation

- Year: 2026
- Venue: arXiv
- Authors: Homaira Huda Shomee; Rochana Chaturvedi; Yangxinyu Xie; Tanwi Mallick
- URL: https://arxiv.org/abs/2602.10017
- BibTeX key (if we add it): score-shomee2026
- Tags: evaluation, rag, reference-free, llm-as-judge, specificity, context-utilization, robustness, relevance

## One-sentence takeaway

SCORE proposes a multi-metric, reference-free evaluation suite for high-stakes/domain RAG QA that explicitly measures specificity, robustness, answer relevance, and context utilization (with readability as an auxiliary metric).

## What problem does it solve?

- Standard reference-based metrics (BLEU/ROUGE-like similarity) and common RAG metrics (faithfulness/factuality, semantic relevance) can miss whether an answer contains *decision-critical*, fine-grained details needed in domain settings (their motivating domain: natural hazards + infrastructure planning).
- Existing reference-free “LLM-as-a-judge” frameworks are often general-purpose and not designed to score domain-specific specificity or evidence usage.

## What is the core method / protocol?

- Build a domain-specific benchmark: 1,412 synthetic question–answer pairs tied to *user profiles* (profession/role, concern type, location, timeline) across 7 hazard types and 40 professional roles.
- Evaluate a RAG answer along multiple dimensions:
  - **Specificity:** decompose answer into atomic claims; extract detail dimensions (hazard, location, timeline, intensity); use multiple LLM evaluators to label each claim/dimension as yes/no/n/a (verifiable vs unsupported) and aggregate with weighted averaging (hazard weighted highest).
  - **Robustness:** generate paraphrases (should be invariant) and semantic perturbations changing hazard/location (should change appropriately); score consistency/sensitivity.
  - **Answer relevance:** “inverse QA” approach—LLM generates candidate questions that the answer would address; compute embedding similarity to original question; add a masking step to reduce spurious lexical overlap from domain entities.
  - **Context utilization:** claim-level leave-one-out dependence—remove individual retrieved claims and measure drop in model answer confidence via teacher-forcing token-prob geometric mean (using proxy open models when generators don’t expose logits); aggregates to global CU and CU_rel.
  - **Readability:** grade-level metrics (auxiliary).

## What are the key metrics?

- Specificity score (weighted across hazard/location/timeline/intensity; uses multi-judge majority voting at claim level).
- Robustness to paraphrase; sensitivity under controlled perturbations (hazard/location changes).
- Answer relevance (masked variant + optional reranking attribution with a reranker model).
- Context utilization (absolute and relative confidence drop under claim removal).
- Readability grade-level metrics.

## What are the main results?

- No single metric captures answer quality well alone; specificity and relevance can disagree (generic-but-relevant vs detailed-but-risky).
- In their experiments, GPT-4o tends to produce more specific (detail-heavy) answers than Gemini 2.5; Gemini can look “more robust” under perturbations partly because it is more generic.
- Context-utilization estimates depend on the proxy evaluator model; LLaMA/Qwen variants give more stable CU behavior than some other small models (per their sensitivity analysis).
- Human evaluation shows substantial subjectivity; agreement is higher for concrete attributes (hazard/location) and lower for timeline/intensity; automated evaluators can approach human–human agreement on some dimensions.

## How is this similar to GALILEO?

- Strong conceptual overlap with *multi-dimensional evaluation* of generated answers beyond single-score metrics.
- Provides concrete operationalizations for (i) robustness to input rephrasings/perturbations and (ii) evidence usage / context dependence—both often relevant when evaluating agentic or RAG-style systems.

## How is this different from GALILEO?

- Target domain is natural hazards + infrastructure decision support; dataset is synthetic and domain-scoped.
- SCORE is primarily an *evaluation framework + benchmark*, not a new generation model/agent.
- Specificity dimensions are tailored (hazard/location/timeline/intensity), which may not transfer directly to other domains without redesign.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO has a domain-agnostic schema for “decision-critical specificity” (or a principled decomposition of what details matter), it could generalize better than SCORE’s hazard-specific dimensions.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently lacks an explicit *context utilization / counterfactual evidence dependence* metric, SCORE’s CU_rel-style leave-one-out approach is a plausible addition.
- If GALILEO uses only relevance/faithfulness-style checks, SCORE highlights why that can miss “generic but safe-looking” answers.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a “specificity” axis in eval, but define domain-appropriate detail dimensions (SCORE uses hazard/location/timeline/intensity).
- [ ] Add robustness tests: paraphrase invariance + semantic perturbations that should change answers (controlled attribute flips).
- [ ] Consider a CU/CU_rel metric (leave-one-out evidence claim removal + confidence drop) to distinguish used vs ignored context.
- [ ] In related work: cite SCORE as a reference-free, multi-metric evaluation framework for high-stakes/domain RAG.

## Quotes / details to potentially cite

- “We propose Score, a multi-perspective evaluation framework … evaluates responses along … specificity, relevance, robustness, context utilization, and readability.” (Intro)
- Dataset: “1,412 domain-specific question–answer pairs spanning 40 professional roles and seven natural hazard types.” (Abstract)
- Specificity weighting (hazard/location/timeline/intensity): 
  - Uses weights α = [0.6, 0.2, 0.1, 0.1] for ordered dimensions hazard, location, timeline, intensity (Sec. 3.1).
