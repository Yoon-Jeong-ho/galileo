# Peak + Accumulation: A Proxy-Level Scoring Formula for Multi-Turn LLM Attack Detection

- Year: 2026
- Venue: arXiv
- Authors: J. Corll (per arXiv submission; full author list not shown on abstract page)
- URL: https://arxiv.org/abs/2602.11247
- BibTeX key (if we add it): corll2026peak
- Tags: multi-turn, prompt-injection, attack-detection, proxy-layer, risk-scoring, safety

## One-sentence takeaway
A simple non-LLM proxy-layer aggregation formula (“peak + accumulation”) turns per-turn pattern scores into a conversation-level multi-turn attack risk score that improves recall at low FPR compared to naive weighted averaging.

## What problem does it solve?
- Detecting *multi-turn* prompt-injection / jailbreak style attacks where malicious intent is spread across turns.
- At the *proxy layer* (i.e., without calling an LLM), aggregate per-turn pattern scores into a single conversation risk score.
- Avoid a flaw in weighted-average aggregation: as turn count increases, it can converge to the per-turn mean and fail to reflect persistence.

## What is the core method / protocol?
- Assume you already have per-turn “pattern” risk scores (rule/pattern matches, categories, etc.).
- Propose an aggregation formula combining:
  - **Peak single-turn risk** (max over turns)
  - **Accumulation / persistence** component (captures sustained suspiciousness across turns; includes a persistence ratio and a tunable parameter ρ)
  - **Category diversity** (more distinct suspicious categories across turns increases risk)
- Motivation analogies: change-point detection (CUSUM), Bayesian belief updating, and security risk alerting.

## What are the key metrics?
- Conversation-level classification: recall, false positive rate (FPR), F1.
- Sensitivity analysis over the persistence parameter (ρ), noting a “phase transition” region.

## What are the main results?
- Evaluated on **10,654 multi-turn conversations**:
  - 588 attacks (from WildJailbreak adversarial prompts)
  - 10,066 benign conversations (from WildChat)
- Reported performance: **90.8% recall at 1.20% FPR**, **F1 = 85.9%**.
- Sensitivity: recall jump (~12 points) around **ρ ≈ 0.4** with negligible FPR increase.
- Releases scoring algorithm + pattern library + evaluation harness (per abstract).

## How is this similar to GALILEO?
- Both care about robust safety / attack detection signals that work across *multi-turn* interactions.
- Emphasis on practical evaluation (large-scale conversation dataset; attack vs benign).
- Proxy-level / system-level scoring perspective aligns with building deployable guardrails.

## How is this different from GALILEO?
- This paper is specifically about **score aggregation** (conversation-level risk from per-turn pattern scores), not necessarily model-based reasoning or more semantically rich detection.
- Operates explicitly **without invoking an LLM** (proxy-only), whereas GALILEO may leverage model signals depending on design.
- Focuses on prompt injection / jailbreak detection datasets (WildJailbreak/WildChat) and pattern libraries.

## Where GALILEO is stronger / cleaner (if true)
- If GALILEO uses richer representations (e.g., semantic/structured features, causal or mechanistic signals, calibrated uncertainty), it may outperform pure proxy aggregation when attackers evade simple patterns.
- GALILEO can potentially unify detection + explanation rather than only scoring.

## Where GALILEO is weaker / needs to improve
- If GALILEO currently aggregates turn-level signals naively (mean/weighted mean/max), it may undercount *persistence*—this paper suggests a concrete failure mode.
- If GALILEO lacks a parameterized persistence mechanism, it may be less controllable in the precision/recall tradeoff.

## Action items for GALILEO (experiments / method / writing)
- [ ] Add a baseline/ablation: **mean vs max vs peak+accumulation-style** aggregation of per-turn suspiciousness.
- [ ] Consider incorporating **persistence ratio** and **category diversity** features into GALILEO’s conversation-level risk scoring.
- [ ] Run a small sensitivity sweep over a persistence hyperparameter (analogous to ρ) and look for regime changes.
- [ ] If we have multi-turn attack benchmarks, report recall at fixed low FPR (proxy-friendly operating point).

## Quotes / details to potentially cite
- “Multi-turn prompt injection attacks distribute malicious intent across multiple conversation turns, exploiting the assumption that each turn is evaluated independently.”
- “We identify a fundamental flaw in the intuitive weighted-average approach: it converges to the per-turn score regardless of turn count…”
- “Evaluated on 10,654 multi-turn conversations … achieves 90.8% recall at 1.20% false positive rate with an F1 of 85.9%.”
- “Sensitivity analysis … reveals a phase transition at rho ~ 0.4 …”
