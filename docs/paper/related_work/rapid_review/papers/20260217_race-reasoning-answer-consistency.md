# Joint Evaluation of Answer and Reasoning Consistency for Hallucination Detection in Large Reasoning Models

- Year: 2025
- Venue: arXiv
- Authors: Changyue Wang; Qingyao Ai; Yiqun Liu
- URL: https://arxiv.org/abs/2506.04832
- BibTeX key (if we add it): wang2025race
- Tags: hallucination-detection, large-reasoning-models, reasoning-consistency, answer-uncertainty, chain-of-thought, black-box-eval

## One-sentence takeaway

RACE is a black-box hallucination detector for large reasoning models that jointly scores **reasoning-trace consistency** and **answer uncertainty/alignment**, catching failures that answer-only consistency misses.

## What problem does it solve?

- LRMs produce long reasoning traces that can be logically inconsistent or unsupported even when final answers look stable across samples.
- Existing sampling-based hallucination detectors (SelfCheckGPT / semantic entropy variants) mostly operate at the **answer level**, and can miss “reasoning hallucinations”.

## What is the core method / protocol?

- Proposes **RACE (Reasoning and Answer Consistency Evaluation)**: sample multiple generations and compute multiple diagnostic signals, then combine them for hallucination detection.
- Includes a “CoT extraction/distillation” step intended to keep only essential reasoning steps (reduce noise from verbose traces).
- Four signals (as described in the arXiv HTML):
  - **Inter-sample reasoning consistency** (do the reasoning traces agree/cohere across samples?)
  - **Entropy-based answer uncertainty** (semantic-entropy-style uncertainty over sampled answers)
  - **Reasoning–answer semantic alignment** (does the reasoning trace actually support/predict the space of sampled answers?)
  - **Internal coherence of reasoning** (heuristic to quantify speculative / internally inconsistent content)

## What are the key metrics?

- Primary: hallucination detection performance vs baselines (paper reports outperforming prior methods across datasets/models; specific metrics not extracted within this rapid pass).
- Component/ablation contributions: shows complementary value of the four modules.

## What are the main results?

- Claims consistent improvement over answer-only uncertainty/consistency baselines on multiple datasets and both LRMs and standard LLMs.
- Key qualitative point: samples can converge on the same final answer while having divergent or flawed reasoning traces; RACE captures this gap.

## How is this similar to GALILEO?

- Shared theme: **robust evaluation beyond surface outputs**—GALILEO cares about belief/stance stability over interaction; RACE argues answer-only signals are insufficient when “hidden” structure (reasoning trace) varies.
- Provides a concrete example of **multi-sample consistency** signals that could be adapted to measure *pressure-induced drift* not only in final claims but in intermediate justification.

## How is this different from GALILEO?

- Not a multi-turn social-pressure / persuasion benchmark; focuses on **hallucination detection** (correctness/groundedness) rather than drift vs revision under interaction.
- Evaluates *within a single prompt* via sampling variability more than *across turns* via sequential perturbations.

## Where GALILEO is stronger / cleaner (if true)

- GALILEO (as framed) can isolate **turn-by-turn dynamics** (time-to-flip, recovery, drift-vs-evidence controls) that RACE does not target.

## Where GALILEO is weaker / needs to improve

- If GALILEO relies heavily on final-answer/stance labels, it may miss cases where models keep the same stance but the **justification degrades** (or vice versa). RACE suggests tracking reasoning-quality consistency as an additional failure axis.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a “reasoning-trace stability” appendix metric: under pressure, does the model’s justification remain coherent/consistent even when the stated answer is stable?
- [ ] Consider a **joint score** for (stance correctness/stability) + (reasoning coherence/alignment), to avoid over-crediting superficially stable answers.
- [ ] Cite RACE as evidence that **answer-only consistency is an incomplete reliability signal** for LRMs.

## Quotes / details to potentially cite

- “Multiple samples may converge on the same final answer but follow divergent or incoherent reasoning paths … hallucinations … remain undetected when evaluation is limited to final answers.” (Intro, arXiv HTML)
- RACE modules: reasoning consistency; answer uncertainty; reasoning–answer alignment; reasoning internal coherence. (Abstract/Intro, arXiv HTML)
