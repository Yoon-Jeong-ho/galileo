# Prompt Repetition Improves Non-Reasoning LLMs

- Year: 2025
- Venue: arXiv
- Authors: Yaniv Leviathan; Matan Kalman; Yossi Matias
- URL: https://arxiv.org/abs/2512.14982
- BibTeX key (if we add it): leviathan2025promptrepetition
- Tags: prompt-repetition, prompt-ordering, prefill, robustness-adjacent, inference

## One-sentence takeaway

A trivial prompting change—concatenating the prompt with itself—reliably boosts accuracy for several frontier LLMs when *reasoning is disabled*, with little-to-no latency increase (except for very long prompts).

## What problem does it solve?

- Many LLM failures in non-reasoning settings are sensitive to prompt token order and attention limitations of causal decoding (e.g., options-first vs question-first in MCQ), leading to avoidable accuracy loss.

## What is the core method / protocol?

- Prompt repetition: transform an input query from `<QUERY>` to `<QUERY><QUERY>`.
- Intuition: by repeating, each token in the query can attend to the “same” information earlier/later in the sequence, partially mitigating order/attention bottlenecks in a causal LM (framed as improving the prompt’s internal attention connectivity).
- Evaluate across 7 models (Gemini 2.0 Flash / Flash Lite; GPT-4o / 4o-mini; Claude 3 Haiku / Claude 3.7 Sonnet; DeepSeek V3) and 7 benchmarks/configurations.
- Special emphasis on multiple-choice formatting variants (question-first vs options-first).

## What are the key metrics?

- Accuracy on standard benchmarks (ARC-Challenge, OpenBookQA, GSM8K, MMLU-Pro, MATH) plus two custom tasks (NameIndex, MiddleMatch).
- Win/loss significance via McNemar test (p < 0.1) on paired predictions.
- Efficiency checks: generated output length + end-to-end latency via provider APIs.

## What are the main results?

- Without reasoning: prompt repetition “wins” 47/70 benchmark-model combinations with 0 losses (per McNemar p < 0.1 criterion), and is generally positive across all tested models.
- Larger gains appear in options-first MCQ formatting (where, without repetition, the model processes answer options without the question context).
- On custom tasks, gains can be dramatic (example reported: Gemini 2.0 Flash-Lite on NameIndex from 21.33% to 97.33%).
- With step-by-step / reasoning-style prompting: effect is neutral to slightly positive.
- Latency/output length: generally unchanged (prefill is parallelizable), with a caveat that very long prompts (and repetition×3) can increase latency for some models.

## How is this similar to GALILEO?

- Both highlight that *re-asking / repetition* can substantially affect outcomes, meaning multi-turn protocols need explicit controls to separate persona/pressure effects from generic re-asking or formatting effects.
- Supports the design intuition behind a neutral re-asking control: repeated exposure alone can change answers (here, usually improving), so repetition effects should be measured rather than assumed negligible.

## How is this different from GALILEO?

- This is not a robustness-to-pressure evaluation; it is a single-turn prompting technique study focused on accuracy improvements under a “no reasoning” setting.
- No notion of conditioning on initial correctness, multi-round flip dynamics, survival/TOF, or recovery after being misled.

## Where GALILEO is stronger / cleaner (if true)

- GALILEO explicitly models multi-turn failure dynamics (time-to-event / survival over rounds) and isolates pressure-persona effects using a matched multi-turn control, rather than altering the prompt itself.

## Where GALILEO is weaker / needs to improve

- GALILEO’s neutral re-asking control (and more generally, any re-asking-based baseline) could be influenced by repetition/formatting tricks; we should be clear that the control is a *drift baseline under the chosen prompting* rather than a universal lower bound.

## Action items for GALILEO (experiments / method / writing)

- [ ] Related work sentence: cite prompt repetition as evidence that repeated / redundant input can systematically shift performance even without new evidence.
- [ ] Consider a small ablation for the control condition: “re-ask with explicit repetition” vs “standard re-ask” (optional; only if it doesn’t explode the matrix), to show that our persona-vs-control deltas are not an artifact of prompt redundancy.

## Quotes / details to potentially cite

- “We propose to repeat the prompt, i.e. transform the input from ‘<QUERY>’ to ‘<QUERY><QUERY>’.”
- “Prompt repetition wins 47 out of 70 tests, with 0 losses.” (non-reasoning setting; McNemar p < 0.1)
- The technique “does not change the format of the generated outputs… enabling simple drop-in deployment in existing systems.”
