# Disentangling the Drivers of LLM Social Conformity: An Uncertainty-Moderated Dual-Process Mechanism

- Year: 2025
- Venue: arXiv
- Authors: Yanan Liu; Qi Cao; Shijin Wang; Zijing Ye; Zimu Wang; Shiyao Zhang
- URL: https://arxiv.org/abs/2508.14918
- BibTeX key (if we add it): liu2025disentangling-llm-conformity
- Tags: conformity, social-influence, uncertainty, information-cascade, bayesian-updating, evaluation

## One-sentence takeaway

LLMs’ conformity in an information-cascade setup is primarily informational (evidence-driven), but under high uncertainty they overweight public/majority signals in a “normative-like” amplification regime.

## What problem does it solve?

- Provides a quantitative way to separate *informational* vs *normative* drivers of LLM social conformity (moving beyond self-reported rationales).
- Tests uncertainty as a moderator that can shift LLM decision behavior (analogous to dual-process accounts in humans).

## What is the core method / protocol?

- Adapts an **information cascade** paradigm from behavioral economics.
- Tasks: three decision-making scenarios (medical, legal, investment).
- Manipulates “information uncertainty” (reported q values differ by scenario).
- Fits a model that estimates decision weights for **private** vs **public** signals (reported as betas / coefficients).
- Evaluates 9 “leading” LLMs (paper does not claim a single-model result).

## What are the key metrics?

- Accuracy vs strength of evidence.
- Confidence vs strength of evidence.
- Estimated weighting of evidence sources:
  - private-signal weight (beta_private)
  - public-signal weight (beta_public)

## What are the main results?

- Across contexts, **informational influence** is the backbone: accuracy + confidence increase with stronger evidence.
- **Uncertainty strongly modulates behavior**:
  - Low-to-medium uncertainty: a *conservative* strategy where models **underweight all evidence sources**.
  - High uncertainty: a regime shift where models still process information but also show **public-signal overweighting** (normative-like amplification), e.g. beta_public > 1.55 vs beta_private ≈ 0.81 (values quoted in the abstract).

## How is this similar to GALILEO?

- Directly relevant if GALILEO studies **multi-agent / majority / consensus effects** on model decisions, especially under uncertainty.
- Offers a concrete operationalization of “social influence” via **public vs private signal weighting**, which maps well to settings where models receive (i) own evidence and (ii) group recommendations.

## How is this different from GALILEO?

- Focus is on *explaining/measuring conformity mechanisms* (informational vs normative-like), not on building a new coordination/aggregation algorithm.
- Uses a behavioral-econ cascade framing rather than (e.g.) learning-to-debate, RLHF-based social objectives, or explicit agent communication protocols.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO provides a controllable mechanism to **prevent herding** or to **calibrate aggregation** (e.g., uncertainty-aware weighting), it can be positioned as a remedy to the high-uncertainty public-signal amplification shown here.

## Where GALILEO is weaker / needs to improve

- If GALILEO relies on majority/public signals, it should explicitly test for the **high-uncertainty amplification** failure mode and show mitigation.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add an “information-cascade-like” evaluation: separate **private evidence** from **public/peer majority** cues; fit/estimate beta_private vs beta_public.
- [ ] Sweep uncertainty (easy/medium/hard; or calibrated ambiguity) to test for a regime shift where public cues become overweighted.
- [ ] In writing: cite this as evidence that LLM conformity is mostly informational but can become *normative-like* at high uncertainty, motivating uncertainty-aware guardrails.

## Quotes / details to potentially cite

- “We evaluated nine leading LLMs across three decision-making scenarios (medical, legal, investment)…”
- “Informational influence underpins the models’ behavior across all contexts…”
- Under high uncertainty: “overweight public signals (beta > 1.55 vs. private beta = 0.81).”
