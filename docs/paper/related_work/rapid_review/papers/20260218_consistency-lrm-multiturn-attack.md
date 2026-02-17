# Consistency of Large Reasoning Models Under Multi-Turn Attacks

- Year: 2026
- Venue: arXiv
- Authors: Yubo Li, Ramayya Krishnan, Rema Padman
- URL: https://arxiv.org/abs/2602.13093
- BibTeX key (if we add it): li2026consistency
- Tags: multi-turn, attack, robustness, reasoning, persuasion, sycophancy, confidence

## One-sentence takeaway

Even frontier “reasoning” LLMs are still systematically persuadable in multi-turn settings; they fail via recognizable social/epistemic modes (self-doubt, conformity, suggestion hijacking, etc.), and confidence-based defenses (CARG) break due to overconfident long reasoning traces.

## What problem does it solve?

- Evaluates whether “large reasoning models” (LRMs; models that produce extended reasoning traces / benefit from inference-time scaling) maintain **answer consistency** when subjected to **multi-turn adversarial pressure** after initially answering correctly.
- Diagnoses *how* and *why* LRMs flip under conversational attacks (beyond single-turn accuracy).

## What is the core method / protocol?

- Multi-turn adversarial evaluation: model answers a factual multiple-choice question, then receives adversarial follow-up turns designed to induce answer flipping.
- Studies 9 frontier reasoning models vs instruction-tuned baselines.
- Attack “tactic” analysis + trajectory analysis to categorize failure modes.
- Examines confidence-calibration and tests Confidence-Aware Response Generation (CARG) variants.

## What are the key metrics?

- Consistency / answer-flip rate under attack (and comparisons vs baselines; reported as effect sizes, e.g., Cohen’s d ranges).
- Confidence–correctness relationship (reported correlation and ROC-AUC).
- Breakdown of failure modes across trajectories.

## What are the main results?

- Reasoning helps but does not solve the problem: **8/9** LRMs show higher consistency than instruction-tuned baselines (reported **d ≈ 0.12–0.40**), yet **all** have vulnerabilities.
- “Misleading suggestions” are broadly effective; “social pressure” effectiveness is model-dependent.
- Identified 5 failure modes:
  - Self-Doubt
  - Social Conformity
  - Suggestion Hijacking
  - Emotional Susceptibility
  - Reasoning Fatigue
  - (Self-Doubt + Social Conformity account for ~50% of failures.)
- Confidence is poorly calibrated / not predictive of correctness in LRMs (reported **r ≈ -0.08**, **ROC-AUC ≈ 0.54**), with systematic overconfidence induced by long reasoning traces.
- CARG (confidence-aware response generation) that can help standard LLMs fails here; notably, “random confidence embedding” can outperform targeted confidence extraction.

## How is this similar to GALILEO?

- Directly targets **multi-turn robustness under pressure** (persuasion / rebuttal / adversarial follow-ups), which is central to GALILEO’s motivation.
- Provides a concrete taxonomy of conversational failure mechanisms that map onto “drift”/“revision” concerns.

## How is this different from GALILEO?

- Focuses on **answer consistency** on factual multi-choice tasks under adversarial follow-ups, rather than (necessarily) richer belief revision norms or long-horizon interaction policies.
- Primarily an **evaluation + diagnosis** paper; does not propose a strong end-to-end training/decoding solution that resolves multi-turn persuasion.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO formalizes *when* belief change is warranted (evidence-aware revision) and separates “social pressure” from “epistemic update,” it can offer a clearer normative target than raw consistency.
- If GALILEO includes interventions/mechanisms beyond confidence heuristics, it can address the paper’s finding that confidence is unreliable in LRMs.

## Where GALILEO is weaker / needs to improve

- If GALILEO evaluations do not include diverse multi-turn adversarial tactics (misleading suggestions, emotional pressure, etc.), it may under-measure real-world vulnerability.
- If GALILEO relies on confidence signals, this paper suggests those signals may be especially misleading for reasoning models.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add this paper as a “reasoning is not robustness” citation in the related-work section.
- [ ] Replicate a subset of their multi-turn attack protocol as a GALILEO stress test (at least “misleading suggestion” and “social pressure” classes).
- [ ] Consider reporting failure-mode breakdowns (self-doubt vs conformity vs hijacking vs fatigue) as an analysis lens for GALILEO.
- [ ] Be cautious about confidence-based defenses/claims; if using confidence, justify calibration or add ablations showing it is not a spurious correlate.

## Quotes / details to potentially cite

- Abstract (paraphrase-able): reasoning gives “meaningful but incomplete robustness,” with failure modes including self-doubt and social conformity; confidence-based defenses (CARG) fail due to overconfidence from extended reasoning traces.
- Reported stats: 8/9 reasoning models > baselines (d ≈ 0.12–0.40); confidence–correctness r ≈ -0.08; ROC-AUC ≈ 0.54.
