# Firm or Fickle? Evaluating Large Language Models Consistency in Sequential Interactions

- Year: 2025
- Venue: ACL 2025 (arXiv)
- Authors: Yubo Li, Yidi Miao, Xueying Ding, Ramayya Krishnan, Rema Padman
- URL: https://arxiv.org/abs/2503.22353 (HTML: https://arxiv.org/html/2503.22353v4)
- BibTeX key (if we add it): li2025firmorfickle
- Tags: multi-turn, consistency, belief-revision, robustness, metrics, benchmark

## One-sentence takeaway

Proposes an evaluation+mitigation package for multi-turn instability—**Position-Weighted Consistency (PWC)** + the **MT-Consistency** benchmark + **Confidence-Aware Response Generation (CARG)**—aimed at keeping LLM answers stable under challenging follow-ups without losing accuracy.

## What problem does it solve?

- In high-stakes multi-turn settings, LLMs often **change answers across follow-ups** (including when follow-ups contain misinformation, social pressure, or reframings), harming trustworthiness.
- Existing metrics often reduce behavior to per-turn correctness and miss **when** instability happens (early vs late) and whether the model **recovers**.

## What is the core method / protocol?

- **PWC (Position-Weighted Consistency):** a consistency metric that weights earlier-turn stability more heavily, and is intended to reflect both early drift and recovery patterns.
- **MT-Consistency dataset/benchmark:** curated multi-domain sequential-interaction test set with “challenging follow-up scenarios” (the paper emphasizes follow-ups that can induce inconsistency).
- **CARG (Confidence-Aware Response Generation):** a decoding/generation framework that uses internal model confidence signals during generation to improve stability (integrates confidence scores to discourage unnecessary answer changes).

## What are the key metrics?

- **PWC** as the headline metric for multi-turn consistency.
- Standard accuracy (paper claims stability gains without sacrificing accuracy).

## What are the main results?

- CARG reportedly **improves response stability** on MT-Consistency while maintaining accuracy (per abstract).
- The paper positions PWC as capturing dynamics that naive flip-rate / unweighted consistency misses.

## How is this similar to GALILEO?

- Same core concern: **multi-turn robustness under pressure** (follow-up challenges causing drift/flip).
- Overlaps with GALILEO’s emphasis that multi-turn behavior requires **trajectory-aware metrics**, not single-turn correctness.

## How is this different from GALILEO?

- Framed as general **consistency** / instability (including knowledge conflicts and misinformation sensitivity) rather than GALILEO’s sharper separation of **evidence-driven belief revision vs pressure-driven drift**.
- Introduces a **confidence-based generation** mitigation (CARG), which GALILEO may treat cautiously because “confidence” can be miscalibrated in precisely the settings that induce drift.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO has explicit controls for “new evidence” vs “pressure only,” that’s cleaner for attributing flips to persuasion vs correction.
- GALILEO can complement PWC with explicit **recovery-after-flip** and “good vs bad flip” accounting (if present).

## Where GALILEO is weaker / needs to improve

- Metric/reporting: PWC is a simple, communicable metric for paper readers; GALILEO should either map to it or justify alternatives.
- Benchmarking: MT-Consistency suggests value in a clearly packaged, multi-domain benchmark artifact; GALILEO may need a similarly legible benchmark release.
- Mitigation baselines: having an explicit “confidence-aware decoding” baseline can be useful, even if it fails in some regimes.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a paragraph comparing GALILEO metrics to **PWC** (and/or include PWC as an auxiliary report), explicitly stating what PWC captures/misses vs drift-vs-revision controls.
- [ ] Consider adding an **MT-Consistency-style operator mix** of follow-ups (misinformation, tone changes, social suggestion, repeated challenge) to show coverage of real sequential interactions.
- [ ] Add a baseline akin to **confidence-aware stabilization** (even if approximate) and test whether it helps/hurts under social-pressure conditions (important for “don’t overtrust confidence” narrative).

## Quotes / details to potentially cite

- Abstract contribution summary: introduces **PWC**, **MT-Consistency**, and **CARG**, claiming improved stability “without sacrificing accuracy.”
- Motivation framing: sequential interactions in high-stakes domains require “consistent and coherent behavior across multiple rounds of user interaction.”
