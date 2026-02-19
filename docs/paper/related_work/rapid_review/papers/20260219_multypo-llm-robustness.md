# Evaluating Robustness of Large Language Models Against Multilingual Typographical Errors

- Year: 2025
- Venue: arXiv (preprint)
- Authors: Yihong Liu, Raoyuan Zhao, Hinrich Schütze, Michael A. Hedderich
- URL: https://arxiv.org/abs/2510.09536
- BibTeX key (if we add it): multypo2025llmrobustness
- Tags: robustness, multilingual, noise, typos, evaluation, prompting

## One-sentence takeaway

Keyboard-layout-aware multilingual typos (MulTypo) cause consistent and sometimes large performance drops across tasks/models, with instruction tuning often increasing clean accuracy but not necessarily robustness under noisy inputs.

## What problem does it solve?

- Existing LLM benchmarks largely assume clean text; robustness to realistic *multilingual* typographical errors is under-measured.
- Prior perturbations are often English-centric and/or unrealistic (edit-distance or character noise without keyboard/typing constraints).

## What is the core method / protocol?

- Proposes **MulTypo**, a multilingual typo generation algorithm grounded in:
  - language-specific keyboard layouts,
  - four classic typo types: replacement / insertion / deletion / transposition,
  - constraints meant to mimic human typing (e.g., transposition primarily across different hands under 10-finger conventions),
  - length-aware word sampling (prob ∝ sqrt(word length)) and position-aware character sampling (errors more likely mid/end),
  - “ignore sets” to avoid corrupting numerals/number-words (to keep evaluation focused on typo robustness rather than changing numeric values).
- Validates naturalness via **human evaluation** comparing MulTypo vs a naive baseline (same operations, no layout constraints).
- Evaluates **18 open-source LLMs** across three families (Gemma, Qwen, OLMo), base + instruction-tuned variants.
- Tasks (as described in the abstract/intro):
  - natural language inference,
  - multi-choice QA,
  - mathematical reasoning,
  - machine translation,
  - plus another downstream task (paper states five total).
- Studies robustness under varying corruption levels and zero-shot vs few-shot prompting.

## What are the key metrics?

- Task-specific standard metrics (not fully enumerated in the arXiv abstract); generally accuracy / exact match for QA & reasoning, and translation quality metrics for MT (paper likely uses BLEU/ChrF/COMET-style metrics).
- Robustness is primarily measured as **performance degradation under controlled typo rates** vs clean inputs.

## What are the main results?

- Typos **consistently degrade** LLM performance.
- Degradation is **stronger for generative + reasoning-heavy tasks**; NLI appears comparatively more robust.
- **Model size is not a guarantee** of robustness.
- **Instruction tuning** improves clean-input results but may **increase brittleness** under typo noise.
- Robustness differs by language/resource level: **high-resource languages** tend to be more robust than low-resource languages.
- MT asymmetry: translation **from English** is more robust than translation **into English**.
- Few-shot prompting (more examples) does **not reliably improve** typo robustness.

## How is this similar to GALILEO?

- If GALILEO targets real-world robustness/reliability of language(-enabled) systems, this work provides a concrete, *realistic noise model* (typos) and an evaluation protocol to stress-test models under natural user input corruption.
- Emphasizes multilingual robustness rather than English-only stress tests.

## How is this different from GALILEO?

- Primarily an **evaluation + corruption generation** paper (MulTypo), not a new modeling/training method.
- Focus is on *typo noise* (keyboard-driven), not broader distribution shifts or adversarial prompt attacks.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO includes training-time robustness methods or principled uncertainty/rejection, it may go beyond this paper’s evaluation emphasis.
- If GALILEO uses task- or domain-specific error models, it may provide tighter alignment to target deployments.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently evaluates mostly on clean text or English-only settings, it likely underestimates real-world failure modes from multilingual user typos.
- If GALILEO lacks a “human-like typo” corruption suite, it may be missing an easy-to-add but high-impact robustness check.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a **MulTypo-style keyboard-aware typo corruption** condition to GALILEO’s evaluation (at multiple typo rates).
- [ ] Report robustness not just as absolute accuracy, but as **relative drop** vs clean (and possibly AUC over corruption rates).
- [ ] Include multilingual breakdowns: high-resource vs low-resource, script families, and directionality effects (e.g., into vs out of English for MT-like tasks).
- [ ] Test whether instruction-tuned variants (or instruction-style prompting) increase brittleness in GALILEO’s setting.
- [ ] Consider training-time augmentation: typo-noise augmentation or contrastive objectives for robustness (citing this paper as motivation).

## Quotes / details to potentially cite

- “To address this gap, we introduce MulTypo, a multilingual typo generation algorithm that simulates human-like errors based on language-specific keyboard layouts and typing behavior.”
- “Instruction tuning improves clean-input performance but may increase brittleness under noise.”
- Key methodological specifics worth citing:
  - four typo types (replacement/insertion/deletion/transposition),
  - word sampling prob ∝ sqrt(length),
  - transposition constrained by (approx.) two-hand typing convention,
  - human evaluation validating naturalness vs naive corruption.
