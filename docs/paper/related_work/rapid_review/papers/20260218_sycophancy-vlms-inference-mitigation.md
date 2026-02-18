# Sycophancy in Vision-Language Models: A Systematic Analysis and an Inference-Time Mitigation Framework

- Year: 2024
- Venue: arXiv (journal version: Neurocomputing 2026)
- Authors: Yunpu Zhao et al.
- URL: https://arxiv.org/abs/2408.11261
- BibTeX key (if we add it): Zhao2024SycophancyVLM
- Tags: sycophancy, vlm, inference-time, mitigation, contrastive-decoding, query-neutralization

## One-sentence takeaway

A systematic study finds LVLMs are broadly vulnerable to leading/biased prompts, and proposes a training-free inference-time pipeline (query neutralization + contrastive decoding + adaptive logit refinement) that reduces sycophancy without hurting neutral performance.

## What problem does it solve?

- Quantifies and characterizes *sycophancy* in large vision-language models: performance drops / instability when user queries contain leading or deceptive framing.
- Provides an inference-time mitigation strategy that does not require retraining or model-specific changes.

## What is the core method / protocol?

- **Evaluation protocol (analysis):** curate “leading queries” (biased variants of otherwise neutral questions) and measure how LVLM predictions shift / degrade across multiple vision-language benchmarks.
- **Mitigation framework (training-free, model-agnostic, inference-time):**
  - **Query neutralizer:** use an external language model to rewrite user queries to remove implicit leading sentiment/bias.
  - **Sycophancy-aware contrastive decoding:** generate (or score) outputs under both leading and neutralized queries, then *contrast* token-level distributions to downweight sycophancy-driven continuations.
  - **Adaptive logits refinement:** refine the contrasted logits with (i) plausibility filtering and (ii) a query sentiment scaling component to keep generations coherent/grounded.

## What are the key metrics?

- Sycophancy susceptibility measured as **performance degradation** and **instability** under leading prompts across tasks/benchmarks.
- Behavioral diagnostics mentioned: **sentiment sensitivity** and **prediction polarity shifts** (answer flips) induced by leading framing.

## What are the main results?

- Across multiple LVLMs and benchmarks, leading prompts cause consistent drops and unstable behavior.
- The proposed inference-time framework **mitigates sycophancy across evaluated models** while **maintaining performance on neutral prompts**.

## How is this similar to GALILEO?

- Directly aligned with the broader theme of **robustness to conversational pressure / misleading context** (here, leading user prompts in multimodal settings).
- Uses a *paired-conditions* idea (leading vs neutralized) reminiscent of “contrastive” robustness evaluation/intervention patterns.

## How is this different from GALILEO?

- Focuses on **vision-language** tasks and *query framing* effects, rather than multi-turn longitudinal drift/pressure protocols (as in many LLM sycophancy papers).
- Mitigation is **decoding-time / logit-space** plus query rewriting, not training-time alignment or dataset-level interventions.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO emphasizes multi-turn protocols, it may better capture **temporal dynamics** (accumulating pressure, recovery) beyond single-query leading effects.

## Where GALILEO is weaker / needs to improve

- Consider extending robustness coverage to **multimodal** settings: leading text can override visual grounding (a distinct failure mode).
- Might need an **inference-time guardrail** story (training-free mitigation) comparable in practicality.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a “leading vs neutralized query” condition for at least one task family (even if text-only), to measure *polarity flips* and *sentiment sensitivity*.
- [ ] Consider a simple inference-time baseline inspired by this paper: dual-run decoding (original vs neutralized) + lightweight contrastive selection.
- [ ] In related work, cite this as **multimodal evidence** that sycophancy is general and that inference-time fixes can work.

## Quotes / details to potentially cite

- Problem framing (abstract): sycophancy in LVLMs = being “unduly influenced by leading or deceptive prompts,” causing “biased outputs and hallucinations.”
- Proposed components (abstract): “query neutralizer” + “sycophancy-aware contrastive decoding” + “adaptive logits refinement” with “plausibility filter” and “query sentiment scaler.”
