# TeleAI-Safety: A comprehensive LLM jailbreaking benchmark towards attacks, defenses, and evaluations

- Year: 2025
- Venue: Pattern Recognition (per arXiv HTML v2 journal field)
- Authors: Xiuyuan Chen, Jian Zhao, Yuxiang He, Yuan Xun, Xinwei Liu, Yanshu Li, Huilin Zhou, Wei Cai, Ziyan Shi, Yuchen Yuan, Tianle Zhang, Chi Zhang, Xuelong Li
- URL: https://arxiv.org/abs/2512.05485
- BibTeX key (if we add it): chen2025teleaisafety
- Tags: jailbreak, multi-turn, attacks, defenses, evaluation, safety-utility

## One-sentence takeaway

TeleAI-Safety is an “all-in-one” modular framework + standardized benchmark for LLM jailbreak safety that unifies **attack**, **defense**, and **evaluation** components and reports safety–utility trade-offs across many models.

## What problem does it solve?

- Existing jailbreak/safety benchmarks are often *imbalanced* (many attacks but few defenses, or vice versa) and/or separate “framework tooling” from “benchmark standardization,” making cross-paper comparisons and end-to-end assessments hard.
- Practitioners need a reproducible harness to mix-and-match attacks/defenses/evaluators while still producing standardized scores.

## What is the core method / protocol?

- Build a **modular evaluation harness** integrating:
  - attacks (arXiv: 19 attack methods; GitHub repo claims 21 in the released codebase)
  - defenses (29 families)
  - evaluation methods (19)
- Curate a standardized **attack corpus**:
  - 342 samples spanning 12 risk categories.
- Run a benchmark over a slate of target models:
  - arXiv abstract: 14 target models;
  - GitHub README: 17 target models (11 closed, 6 open) — suggests the released benchmark may have expanded past the paper’s initial set.
- The paper highlights two “self-developed” methods integrated into the suite:
  - **Morpheus**: self-evolving, metacognitive multi-round attack agent (adaptive multi-turn jailbreak).
  - **RADAR**: multi-agent debate-based safety evaluation method.

## What are the key metrics?

- Primary: **Attack Success Rate (ASR)** (explicitly emphasized in the released repo figures).
- The benchmark is framed as jointly tracking **safety** and **utility** (trade-off), though the arXiv snippet does not expose the exact utility metric definition (likely a benign-task helpfulness/quality score and/or refusal/over-refusal accounting).
- Breakdown views include:
  - ASR across black-box vs white-box settings (per repo results figures).
  - Safety performance across risk categories (per repo “radar” plot).

## What are the main results?

- Across evaluated models, the benchmark surfaces:
  - systematic vulnerabilities (jailbreak success remains non-trivial under many attack families),
  - model-specific failure modes,
  - non-trivial **safety–utility** trade-offs where defenses may reduce ASR but can also degrade helpfulness.

(Details are reported in the benchmark’s aggregated ASR plots; this rapid note focuses on the framework/protocol contribution rather than exact numbers.)

## How is this similar to GALILEO?

- Same *multi-turn robustness* spirit: evaluates failures that unfold over interaction (especially via adaptive/multi-round attack agents).
- Emphasizes that evaluation must consider **protocol + metrics**, not just one-shot success rates.
- Explicitly frames **safety vs utility** trade-offs, aligning with GALILEO’s preference to avoid “refuse everything” degenerate solutions.

## How is this different from GALILEO?

- Focus is jailbreak safety broadly (harmful instruction elicitation) rather than **belief drift / social-pressure-induced flips** with neutral controls.
- Central outcome is often ASR, not **time-to-failure / survival** or **recovery after flip**.
- Does not (in the visible excerpt) isolate “evidence-driven revision” vs “pressure-driven drift” with paired controls.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO reports **time-to-event**, hazard/survival curves, and **recovery dynamics**, it offers a trajectory-sensitive view that ASR alone can hide.
- If GALILEO includes explicit neutral-vs-pressure controls, it can make a cleaner causal claim about drift vs revision than generic jailbreak suites.

## Where GALILEO is weaker / needs to improve

- TeleAI-Safety shows the value of a *single integrated harness* with a large library of attacks/defenses/evaluators; GALILEO may need a clearer “plug-in” story (how easy is it to add new pressure operators / judges / defenses).
- GALILEO could benefit from more explicit **safety–utility dashboards** and category-wise breakdowns (analogous to TeleAI-Safety’s risk-category reports).

## Action items for GALILEO (experiments / method / writing)

- [ ] Writing: cite TeleAI-Safety as evidence that the field is moving toward **integrated framework+benchmark** designs; position GALILEO as the specialized benchmark for *social-pressure belief stability* with drift-vs-revision controls.
- [ ] Experiments: consider adding an “operator library” section (pressure operators + recovery interventions) and documenting a standardized evaluation harness API.
- [ ] Reporting: add a compact **safety–utility** summary panel (even if the axes are different) to prevent “wins” that come from over-refusal.

## Quotes / details to potentially cite

- “TeleAI-Safety … integrates a broad collection of 19 attack methods … 29 defense methods, and 19 evaluation methods … [with] 342 samples spanning 12 distinct risk categories … [evaluations] across 14 target models.” (arXiv abstract)
- Repo summary: “integrates 21 attack methods … 29 families of defenses … 19 complementary evaluation protocols … 342 exemplars … benchmarking on 17 target models.” (GitHub README)
