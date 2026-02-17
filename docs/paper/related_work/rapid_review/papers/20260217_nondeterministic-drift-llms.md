# Quantifying non deterministic drift in large language models

- Year: 2026
- Venue: arXiv
- Authors: Claire Nicholson
- URL: https://arxiv.org/abs/2601.19934
- BibTeX key (if we add it): Nicholson2026NondeterministicDrift
- Tags: nondeterminism, drift, repeated-runs, reproducibility, reliability, evaluation

## One-sentence takeaway

Even with temperature=0, repeated runs of the same prompt can yield measurably different outputs; this paper proposes simple lexical/length metrics and a systematic repeated-run protocol to quantify that baseline “operator-free” variability across models, prompt types, and prompting modes.

## What problem does it solve?

- Provides an empirical *baseline* for output variability (“behavioural drift”) when issuing identical prompts multiple times, highlighting that “deterministic” decoding settings do not guarantee reproducibility.
- Helps practitioners reason about whether observed changes are meaningful vs. within expected nondeterministic noise.

## What is the core method / protocol?

- Repeated-run measurement study (explicitly *no* stabilization/control prompting).
- Models:
  - gpt-4o-mini (API served)
  - llama3.1-8b (local, vLLM)
- Prompting modes:
  - Exact repeats (same prompt run many times)
  - Perturbed inputs (minor lexical/synonym changes)
  - Reuse mode (feed previous answer back into next prompt)
- Temperatures: 0.0 and 0.7.
- Prompt categories (5): underspecified summaries, underspecified advice, light constraints, style prompts, hard constraints.
- Repetitions: up to ~30 runs per prompt (paper mentions 30 for a “gapfill” set and 20 for a smaller battery; errors/refusals excluded).

## What are the key metrics?

- Unique output fraction (how often the completion is distinct across repeats).
- Lexical similarity between outputs (token/word overlap-style measure; paper frames it as lexical similarity).
- Word count / length statistics and variation.

## What are the main results?

- Nondeterminism persists even at temperature = 0.0.
- Variability patterns differ by:
  - Model (API-served vs local open-weight)
  - Prompt category
  - Prompting mode (exact vs perturbed vs reuse)
- The paper emphasizes that these are *baseline* measurements meant to be compared against future drift-mitigation / stabilization techniques.

## How is this similar to GALILEO?

- Shared interest: quantifying instability/robustness phenomena in LLM behavior via repeated interactions.
- Aligns with GALILEO’s need for clear baselines and “noise floors” when interpreting multi-turn degradation or turn-of-failure style metrics.

## How is this different from GALILEO?

- This paper focuses on *repeated-run nondeterminism* (same prompt, multiple runs), not adversarial multi-turn attacks or conversational robustness per se.
- Metrics are primarily lexical/length-based, not task-success, truthfulness, or time-to-failure under adversarial pressure.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO uses outcome/behavioral metrics tied to correctness/safety (and survival/time-to-failure), it captures more semantically meaningful failure than pure lexical drift.
- Multi-turn protocols better match real “conversation over time” reliability needs.

## Where GALILEO is weaker / needs to improve

- GALILEO may under-account for *infrastructure/serving nondeterminism* as a confound (especially when comparing across providers or runs).
- Might need an explicit “baseline repeated-run variability” section or control experiment when reporting multi-turn instability.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a short baseline experiment: repeated identical runs (temp=0) on a subset of prompts/tasks to estimate a reproducibility noise floor.
- [ ] When reporting multi-turn degradation, explicitly separate (a) within-run multi-turn effects from (b) across-run nondeterminism.
- [ ] Consider tracking both lexical variance and semantic/outcome variance to show when nondeterminism actually changes conclusions.

## Quotes / details to potentially cite

- Defines “baseline behavioural drift” as output variability when the same prompt is run repeatedly under an “operator-free” regime (no stabilizing instructions).
- Evaluates gpt-4o-mini and llama3.1-8b across five prompt categories, three prompting modes (exact repeats, perturbed inputs, reuse), at temperatures 0.0 and 0.7.
- Drift measured with unique output fractions, lexical similarity, and word count statistics; reports nondeterminism even at temperature 0.0.
