# PACIFIC: a framework for generating benchmarks to check Precise Automatically Checked Instruction Following In Code

- Year: 2026 (arXiv Dec 2025)
- Venue: AI-SQE 2026 workshop (accepted; ACM)
- Authors: Itay Dreyfuss; Antonio Abu Nassar; Samuel Ackerman; Axel Ben David; Eitan Farchi; Rami Katan; Orna Raz; Marcel Zalmanovici (IBM Research)
- URL: https://arxiv.org/abs/2512.10713
- BibTeX key (if we add it): dreyfuss2025pacific
- Tags: instruction-following, code, benchmarking, deterministic-eval, dry-running, contamination

## One-sentence takeaway

PACIFIC is a benchmark-generation framework for code instruction-following that yields deterministic, automatically checkable expected outputs and allows explicit difficulty control (instruction-chain length + target output length).

## What problem does it solve?

- Lack of code-specific instruction-following benchmarks with **deterministic** (non-LLM-judge) evaluation.
- Need to evaluate an LLM’s intrinsic ability to **dry-run** code / apply sequential transformations without tool execution.
- Need for **contamination-resilient** benchmarks via easy generation of novel variants.

## What is the core method / protocol?

- Define an **instruction pool** of small, dry-runnable transformations (number/string → number/string), each available as:
  - natural-language instruction text, and
  - reference implementations (multiple languages mentioned: Python/Java/C++).
- Generate benchmark samples by composing a **pipeline** (sequence) of instructions applied to an initial input.
- Control difficulty via benchmark-generation parameters, especially:
  - number of instructions per sample (longer reasoning chain), and
  - target output length (characters for strings; bits for numbers), with length-aware selection.
- Compute expected outputs by executing reference implementations; evaluate model outputs by **simple output comparison** (and by parsing intermediate results when requested).

## What are the key metrics?

- Prompt-level accuracy: fraction of samples where **all** instructions are followed correctly.
- Instruction-level accuracy: average fraction of correctly executed instructions across all samples.
- (Also mentions categorizing invalid/incomplete outputs: insufficient answers, type mismatches, etc.)

## What are the main results?

- PACIFIC can generate increasingly challenging benchmarks (by scaling chain length / output length) that better differentiate model capabilities.
- Even strong contemporary LLMs struggle on certain compositions, especially as difficulty increases.

## How is this similar to GALILEO?

- Both emphasize **ground-truth / automatically checkable** evaluation signals (avoid subjective judging).
- Both focus on **sequential, multi-step behavior** (PACIFIC: instruction chains; GALILEO: multi-turn pressure/control).
- Both benefit from **contamination robustness** via generating many variants.

## How is this different from GALILEO?

- PACIFIC targets **code dry-running + instruction-following**; GALILEO targets **multi-turn persona pressure robustness** (survival / turn-of-failure / recovery) on ground-truth tasks.
- PACIFIC’s main knobs are chain length and output length; GALILEO’s main knobs are **interaction dynamics** (persona types, rounds, control vs pressure, recovery protocol).

## Where GALILEO is stronger / cleaner (if true)

- Explicitly disentangles pressure-driven failures from generic drift via a **neutral re-asking control**.
- Measures dynamics beyond final accuracy (survival curves, TOF distributions, recovery@flip), which PACIFIC’s static accuracy metrics don’t capture.

## Where GALILEO is weaker / needs to improve

- GALILEO could benefit from PACIFIC-style **procedural generation** to create many difficulty-controlled variants (reducing overfitting/contamination concerns for fixed task sets).
- GALILEO’s tasks could adopt more explicit **difficulty scalars** akin to (instruction chain length, target output length).

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding a “generated-task” track: automatically generate ground-truth tasks with explicit difficulty controls (e.g., compose transformations / constraints), to complement existing datasets.
- [ ] Add an additional reporting view that mirrors PACIFIC: prompt-level vs step-level (per-turn/per-step) correctness, to better localize where pressure causes failures.
- [ ] Related work positioning: cite PACIFIC as code-domain deterministic instruction-following benchmark generation; contrast with GALILEO’s focus on multi-turn social pressure and drift controls.

## Quotes / details to potentially cite

- PACIFIC aims to “isolate and evaluate the LLM’s intrinsic ability to reason through code behavior step-by-step without execution (dry running) and to follow instructions” (abstract).
- Deterministic evaluation via “simple output comparisons” enabled by “clearly defined expected outputs” (abstract).
- Difficulty dimensions highlighted: (1) number of instructions in each sample; (2) expected output length (intro).
