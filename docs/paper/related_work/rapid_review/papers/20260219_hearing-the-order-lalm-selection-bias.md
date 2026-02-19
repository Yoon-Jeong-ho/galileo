# Hearing the Order: Investigating Selection Bias in Large Audio-Language Models

- Year: 2025
- Venue: arXiv (submitted to ICASSP 2026)
- Authors: Yu-Xiang Lin et al. (first two authors contributed equally)
- URL: https://arxiv.org/abs/2510.00628
- BibTeX key (if we add it): lin2025hearing
- Tags: order-effects, selection-bias, audio-language, evaluation, mcq

## One-sentence takeaway

Large audio-language models show strong multiple-choice option-order (position) bias—up to ~24% accuracy swing—so permutation-based evaluation is often needed for reliable ranking.

## What problem does it solve?

- Tests whether LALMs’ multiple-choice predictions depend on the *order / position* of answer options (selection/position bias), which can make benchmark scores and model rankings unreliable.
- Extends prior “option order bias” work from text LLMs / VLMs to the audio-language setting.

## What is the core method / protocol?

- Evaluate 6 LALMs on MCQ benchmarks (text) and “spoken” counterparts (audio) created by converting questions+options to TTS.
- Benchmarks: MMAU (test-mini), MMAR, MMLU; plus SPEECH-MMAU / SPEECH-MMAR / SPEECH-MMLU.
- Construct controlled variants by *reassigning the correct answer* to each fixed position (A/B/C/D), shuffling remaining options, then measure how accuracy changes by position.
- Analyze bias with:
  - Accuracy and Δ accuracy relative to the original ordering.
  - Relative Standard Deviation (RSD) across positions.
  - Choice KL divergence (CKLD) between predicted-choice distribution and ground-truth label distribution.
- Also tests whether including option identifiers (A/B/C/D) helps.
- Studies permutation-based mitigation strategies (treat each shuffled option order as an independent input; aggregate), analogous to permutation/self-consistency style evaluation.

## What are the key metrics?

- Accuracy; Δ accuracy under controlled “correct-answer-at-position-{A,B,C,D}” conditions.
- RSD across positions (stability of accuracy vs position).
- CKLD (selection distribution skew vs label distribution).

## What are the main results?

- Selection bias is pervasive: *all* evaluated LALMs exhibit accuracy fluctuations when the correct answer is forced into different positions.
- Magnitude can be large (reported up to ~24% max fluctuation for Phi-4-Multimodal in their setting), and can even change model rankings.
- Bias direction differs by model and modality (examples reported):
  - Phi-4-Multimodal tends to favor A and avoid D.
  - Qwen2.5-Omni (3B/7B) tends to avoid D and relatively prefer A.
  - Gemini-2.0-Flash generally prefers D.
  - Voxtral variants show differing patterns; sometimes modality-dependent.
- Adding identifiers (A/B/C/D) can improve accuracy but does not reliably mitigate bias.
- Permutation-based strategies generally reduce bias and can improve reliability, at the cost of extra test-time compute.

## How is this similar to GALILEO?

- Directly relevant to any GALILEO evaluation that uses discrete candidate sets (MCQ-style prompts, selection among proposed outputs, or option lists): order sensitivity can masquerade as capability.
- Supports the motivation for “robust evaluation under perturbations / permutations” rather than single-shot scoring.

## How is this different from GALILEO?

- Focus is on *LALMs* and MCQ benchmarking; not on GALILEO’s specific task structure.
- Uses controlled reassignment of correct answers to positions, rather than GALILEO-specific intervention/ablation protocols.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO reports permutation-robust metrics (averaging over multiple candidate orderings) or avoids MCQ-style selection, it can claim more reliable evaluation than typical single-order MCQ benchmarks.

## Where GALILEO is weaker / needs to improve

- If any GALILEO experiments present ordered candidate lists (or ordered tool/action menus) without randomization, results may inherit similar position bias.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add an “order robustness” check: randomly permute candidate/output order (or option labels) and report mean±std accuracy / success rate.
- [ ] If permutation is expensive, adopt a small fixed set of permutations (e.g., cyclic) and aggregate (majority vote / average logprob).
- [ ] In the paper, explicitly discuss selection/position bias as an evaluation pitfall; cite this as audio-language evidence that the issue generalizes beyond text.

## Quotes / details to potentially cite

- “Shuffling the order of answer options can cause performance fluctuations of up to 24% and even change model rankings…” (abstract)
- Datasets used: MMAU, MMAR, MMLU and spoken counterparts.
- Models: Gemini-2.0-Flash; Phi-4-Multimodal; Qwen2.5-Omni-3B/7B; Voxtral-Mini-3B; Voxtral-Small-24B.
