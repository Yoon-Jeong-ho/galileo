# Benchmarking Correctness and Security in Multi-Turn Code Generation

- Year: 2025
- Venue: arXiv
- Authors: Ruchit Rawal; Jeffrey Yang Fan Chiang; Chihao Shen; Jeffery Siyuan Tian; Aastha Mahajan; Tom Goldstein; Yizheng Chen
- URL: https://arxiv.org/abs/2510.13859
- BibTeX key (if we add it): Rawal2025MTSec
- Tags: multi-turn, code-generation, security, correctness, benchmark, code-diff

## One-sentence takeaway

MT-Sec is (to the authors’ knowledge) the first benchmark that jointly measures *correctness + security* for **multi-turn** code generation, showing a consistent ~20–27% drop in “correct & secure” outcomes when moving from single-turn to multi-turn coding workflows.

## What problem does it solve?

- Existing correctness/security benchmarks for coding LLMs are mostly **single-turn**, missing the iterative, conversational nature of real development.
- Security evaluation can be especially misleading if a model “passes tests” but introduces vulnerabilities during iterative edits.

## What is the core method / protocol?

- Introduce **MT-Sec**, a benchmark built via a **synthetic data pipeline** that converts existing *single-turn* coding tasks into **semantically aligned multi-turn interaction sequences**.
- Key design choice: preserve/reuse the **original test suites** from the source tasks, while adding multi-turn structure to better mimic real workflows.
- Evaluate:
  - **Full-program generation** in multi-turn settings.
  - **Multi-turn code-diff generation** (iterative patching), argued as practically important and under-explored.
- Run experiments across **32 open + closed models** and **three agent scaffoldings**.

## What are the key metrics?

- Primary: rate of outputs that are simultaneously **functionally correct** (e.g., passes tests) and **secure** (“correct and secure” as the headline metric).
- Additional: performance degradation from single-turn → multi-turn; for code-diff setting, rates of **functionally incorrect** and **insecure** outputs.

## What are the main results?

- A consistent **20–27% absolute drop** in “correct and secure” outputs when moving from single-turn to multi-turn, including for strong SOTA models.
- **Multi-turn code-diff generation** is even harder: worse performance and increased rates of incorrect/insecure outputs.
- Agent scaffoldings improve single-turn performance, but are **less effective** in the multi-turn evaluation setting.

## How is this similar to GALILEO?

- Same broad motivation: **single-turn evaluations can overestimate robustness**; multi-turn interactions reveal failure modes that matter in practice.
- Emphasizes *trajectory / interaction protocol* as part of the benchmark definition (not just a static prompt).

## How is this different from GALILEO?

- Domain: **code generation** (plus explicit security), rather than GALILEO’s focus on multi-turn behavioral robustness phenomena.
- Objective: measures *correctness+security* rather than belief/stance drift (or “return to truth”) style outcomes.
- Construction: relies on a **synthetic transformation** of single-turn datasets into multi-turn sequences (with test-suite reuse).

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO uses more “behavioral” or adversarial multi-turn protocols with explicit controls, it may better isolate *why* degradation happens (vs. simply measuring that it does).
- GALILEO can incorporate richer failure-time / turn-of-failure metrics; MT-Sec headline result is primarily a **drop rate** between single vs multi-turn.

## Where GALILEO is weaker / needs to improve

- MT-Sec highlights the value of **joint metrics** (e.g., correctness *and* security). If GALILEO currently reports a single dimension, consider multi-objective scoring.
- Including a **code-diff / patch-based** multi-turn setting is a strong “realistic workflow” move that GALILEO might emulate in its own domain (e.g., incremental revisions).

## Action items for GALILEO (experiments / method / writing)

- [ ] In related work, cite MT-Sec as evidence that **multi-turn evaluation reveals large drops** even when single-turn performance looks strong.
- [ ] Consider adding an “edit/patch” variant of GALILEO tasks (multi-turn *diffs* rather than regenerate-from-scratch), mirroring MT-Sec’s code-diff evaluation idea.
- [ ] If applicable, add a “joint objective” metric (analogous to “correct & secure”), e.g., “truthful & safe” or “stable & calibrated”.

## Quotes / details to potentially cite

- “We introduce MT-Sec, the first benchmark to systematically evaluate both correctness and security in multi-turn coding scenarios.”
- “We observe a consistent 20–27% drop in ‘correct and secure’ outputs from single-turn to multi-turn settings.”
- “Beyond full-program generation, we also evaluate models on multi-turn code-diff generation … and find that models perform worse here …”
