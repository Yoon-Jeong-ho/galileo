# Exploring LLM Reasoning Through Controlled Prompt Variations

- Year: 2025
- Venue: arXiv preprint
- Authors: Giannis Chatziveroglou; Richard Yun; Maura Kelleher
- URL: https://arxiv.org/abs/2504.02111
- BibTeX key (if we add it): chatziveroglou2025promptvariations
- Tags: reasoning-robustness, prompt-perturbation, gsm8k, distractors, evaluation

## One-sentence takeaway

Systematic “noise” added to GSM8K prompts (especially irrelevant context) can sharply degrade LLM math accuracy, with drops not well explained by step-count difficulty or model size.

## What problem does it solve?

- Quantifies how fragile current LLM reasoning performance is to controlled prompt perturbations that resemble real-world “messy” inputs (irrelevant context, misleading instructions, extra-but-related details).

## What is the core method / protocol?

- Use GSM8K word problems as a controlled reasoning benchmark.
- Construct four perturbation categories applied to prompts:
  - Irrelevant context (information-rich distractor text within the context window)
  - Pathological instructions / deceptive cues
  - Factually relevant but non-essential context
  - Combination of (pathological + relevant) ("combo")
- Evaluate 13 models (mix of open and closed) on baseline vs perturbed prompts; compare accuracy deltas.
- Qualitatively inspect generations for changes in reasoning behavior (incl. “chain-of-thought-like” behavior triggered without explicit CoT prompting).

## What are the key metrics?

- Primary: GSM8K accuracy / number (or percent) of correct answers under each perturbation vs baseline.
- Secondary analyses mentioned: relationship of regression magnitude to (a) reasoning complexity proxied by step count and (b) model size.

## What are the main results?

- Irrelevant context inside the context window significantly degrades performance across models.
- Performance drops are relatively insensitive to “reasoning complexity” (step-count proxy).
- Regressions are not strictly correlated with model size.
- Some perturbations can inadvertently trigger more explicit, chain-of-thought-like reasoning even without requesting it.

## How is this similar to GALILEO?

- Both are concerned with reliable reasoning behavior under realistic input conditions (noise, distractions, misleading cues) rather than only clean benchmark prompts.
- Suggests an evaluation axis for GALILEO: robustness to context pollution / instruction attacks / extra context.

## How is this different from GALILEO?

- This is primarily an evaluation study on math word problems (GSM8K), not a new reasoning method.
- Focuses on prompt perturbations rather than model-side algorithmic changes.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO includes explicit mechanisms for relevance filtering / structured reasoning / tool use, it may be less sensitive to irrelevant-context overload than purely prompt-driven baselines.

## Where GALILEO is weaker / needs to improve

- If GALILEO relies on long-context inputs, this paper suggests a concrete failure mode: accuracy can degrade simply from adding plausible but irrelevant context.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a “prompt perturbation robustness” section to evaluation: inject irrelevant context, misleading instructions, and relevant-nonessential details; report accuracy deltas.
- [ ] Include an ablation measuring robustness vs context length and vs distractor entropy (e.g., random wiki vs domain-related prose).
- [ ] If GALILEO has a relevance-selection step, explicitly test whether it mitigates irrelevant-context regressions.

## Quotes / details to potentially cite

- Perturbation taxonomy used: “irrelevant context”, “pathological instructions”, “factually relevant but non-essential context”, and “combo”.
- Main observation: adding irrelevant context “significantly degrades performance”, and the regressions are not well predicted by step-count difficulty or model size (per abstract).
