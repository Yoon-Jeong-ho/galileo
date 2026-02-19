# VAL-Bench: Belief Consistency as a measure for Value Alignment in Language Models

- Year: 2026 (arXiv v3 Jan 2026)
- Venue: arXiv
- Authors: Aman Gupta; Denny O’Shea; Fazl Barez
- URL: https://arxiv.org/abs/2510.05465
- BibTeX key (if we add it): valbench2026gupta
- Tags: values, belief-consistency, framing, prompt-pairs, llm-as-a-judge, refusal, robustness

## One-sentence takeaway

VAL-Bench proposes a large-scale benchmark of *paired, oppositely framed* value-laden prompts (115K pairs) and uses an LLM judge to score whether models maintain a consistent stance vs contradict themselves, highlighting large cross-model variance and the role of refusals.

## What problem does it solve?

- Existing “value alignment” benchmarks often rely on hypothetical/common-sense scenarios and do not directly test whether a model’s *expressed value stance* is stable under *opposing framings* of the same controversial issue.
- Inconsistent expressed beliefs across framing can cause “epistemic harm” by making user beliefs dependent on how questions are asked.

## What is the core method / protocol?

- Dataset construction (Wikipedia-based):
  - Mine Wikipedia sections likely to be controversial (e.g., “Criticism”, “Scandals”).
  - Filter sections to keep “divergent issues”.
  - Use an LLM (paper mentions Gemma-3-27B-it) to generate *two prompts* from the same section, each starting with “Explain why …”, one intended to elicit a “for” stance and the other an “against” stance.
  - Result: ~114,745 prompt pairs.
- Evaluation:
  - For each prompt pair, query a target model for responses to the “for” and “against” prompts.
  - Use an LLM-as-a-judge (validated against human labels) to score agreement/consistency between the two responses on an ordinal scale (paper defines sigma in {-2,-1,0,1,2}).
  - Explicitly account for refusal behavior: if one response refuses, score is adjusted upward (treating refusal as a way to preserve consistency).
  - Also categorize behavior into “value preference” vs “value indifference” (e.g., hedging both sides / refusing both sides).

## What are the key metrics?

- Pairwise agreement / consistency score (ordinal; complete disagreement to complete agreement).
- Value preference vs value indifference indicator (binary in the paper’s formulation).
- Refusal rate and “no-information” response rate are tracked as auxiliaries; scoring is adjusted when refusals occur.

## What are the main results?

- Consistency varies widely across models (paper summary: roughly ~10% to ~80% depending on model).
- Claude-family models are reported as substantially more consistent than GPT-family models on this benchmark, with refusal strategies contributing strongly to the observed differences.
- The paper emphasizes trade-offs: higher “consistency” may be achieved via refusals / reduced expressivity.

## How is this similar to GALILEO?

- Both evaluate *stability* of a model’s outputs under systematically varied prompts, and treat “instability” as a reliability/alignment-relevant failure mode.
- Both emphasize protocolized measurement rather than anecdotal examples.

## How is this different from GALILEO?

- VAL-Bench is primarily **single-shot per prompt** (two separate prompts per issue) rather than a multi-turn dialogue with sequential pressure.
- There is **no ground-truth answer** in the factual sense; it measures *stance consistency* on controversial value issues, not truth maintenance.
- It does not target *persona pressure* or “turn-of-failure / survival over turns”; instead it targets *framing-induced contradictions*.
- The scoring explicitly *credits refusals* as consistency-preserving, whereas GALILEO’s framing is closer to “maintain the correct answer under pressure” (refusal may be treated differently depending on task design).

## Where GALILEO is stronger / cleaner (if true)

- Ground-truth tasks: GALILEO can separate “correctness maintenance” from mere consistency, and can cleanly interpret flips as *wrong-answer adoption*.
- Multi-turn dynamics: GALILEO directly measures time-to-failure / survival curves / recovery after flip under sustained pressure + drift control.

## Where GALILEO is weaker / needs to improve

- Value/factual boundary: GALILEO’s claims likely shouldn’t generalize to value-laden stance consistency; VAL-Bench highlights a parallel but distinct axis.
- Handling refusals: VAL-Bench surfaces that “consistency” can be inflated by refusal; GALILEO should be explicit about how refusals are treated in survival/TOF/recovery metrics.

## Action items for GALILEO (experiments / method / writing)

- [ ] Related work positioning: cite VAL-Bench as “paired opposing framings / consistency under framing”, and contrast with GALILEO’s “ground-truth + multi-turn pressure + survival/TOF/recovery + neutral re-asking control”.
- [ ] Add a short paragraph in related work/discussion on **refusal vs consistency**: consistency benchmarks may conflate “being consistent” with “refusing to answer”; clarify where GALILEO sits.
- [ ] Consider (optional) a small ablation: treat “refusal” as a separate absorbing state vs incorrect answer, and report how often pressure causes refusal vs flip.

## Quotes / details to potentially cite

- Dataset scale and source: “VAL-Bench consists of 115K pairs of prompts … extracted from Wikipedia.”
- Evaluation framing: “paired prompts designed to elicit opposing stances on a controversial issue”.
- Core claim: inconsistency risks “epistemic harm by making user beliefs dependent on how questions are framed”.
- Reported range: consistency rates “ranging from ~10% to ~80%” across models.
