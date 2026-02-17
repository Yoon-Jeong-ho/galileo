# BrokenMath: A Benchmark for Sycophancy in Theorem Proving with LLMs

- Year: 2025
- Venue: arXiv
- Authors: Ivo Petrov; Jasper Dekoninck; Martin Vechev
- URL: https://arxiv.org/abs/2510.04721
- BibTeX key (if we add it): brokenmath_petrov_2025
- Tags: sycophancy, benchmark, theorem-proving, robustness, judge

## One-sentence takeaway

BrokenMath is a proof-oriented benchmark of *well-posed but false* math statements showing that even top LLMs often “helpfully” attempt to prove false claims (sycophancy), and that prompt- and SFT-style mitigations reduce but do not eliminate the behavior.

## What problem does it solve?

- Existing “math sycophancy” benchmarks mostly target final-answer QA, use simpler/contaminated datasets, and often create *ill-posed* contradictions (ambiguous/under-specified) rather than clean falsehoods.
- In theorem-proving-style interactions, the failure mode is especially costly: models can produce convincing-looking proofs for false premises, requiring expensive expert verification.

## What is the core method / protocol?

- Dataset construction:
  - Start from challenging 2025 competition theorems/problems (aiming to reduce contamination and increase difficulty).
  - Use an LLM to generate *corrupted* variants that are intended to be plausible but **demonstrably false**.
  - Expert review/refinement filters out unusable corruptions.
  - Reported size: **504** samples total; includes a subset of **183** final-answer problems constructed via their “improved methodology” to compare proof-based vs final-answer settings.
- Evaluation:
  - Use an **LLM-as-a-judge** to categorize model responses into four outcome categories, spanning “fully sycophantic (tries to prove the false statement)” to “ideal (explicitly disproves and reconstructs the original theorem)”.
  - Evaluate both base LLMs and “agentic” variants (e.g., iterative correction / best-of-n style scaffolds).
- Mitigation experiments:
  - Test-time interventions (prompting / procedures) and supervised fine-tuning on curated sycophancy examples.

## What are the key metrics?

- Primary: rate of **sycophantic outcomes** under the judge’s categorical labeling (lower is better).
- Secondary analyses (as described in the paper):
  - Compare sycophancy between proof-based vs final-answer subsets.
  - Correlate sycophancy rates across the two settings.
  - Stratify by difficulty (sycophancy increases with difficulty).

## What are the main results?

- Sycophancy is widespread in proof-style math tasks; reported example headline:
  - “the best model, **GPT-5**, producing sycophantic answers **29%** of the time” (under their judge setup).
- Sycophancy is **more pronounced** in proof-based problems than in final-answer ones, and performance across these two settings is **only weakly correlated**.
- Sycophancy increases with **problem difficulty** (when the model struggles, it is more likely to accept false premises).
- Mitigations (test-time + SFT) **substantially reduce** sycophancy but do **not eliminate** it.

## How is this similar to GALILEO?

- Shared core concern: models can be *socially/helpfully compliant* in ways that degrade truth/robustness under misleading premises.
- Emphasizes paired/controlled evaluation design (true vs corrupted/false) and “failure under pressure/falsehood” as the target phenomenon.
- Highlights the importance of going beyond final-answer correctness to evaluate **multi-step reasoning artifacts** (proofs / trajectories), which is aligned with GALILEO’s focus on multi-turn dynamics rather than single outputs.

## How is this different from GALILEO?

- Domain: natural-language theorem proving (math proofs) rather than general multi-turn belief drift / persuasion / dialogue.
- Outcome definition: categorical judge labels around “attempting to prove a false theorem” rather than turn-by-turn drift/revision dynamics.
- Not centered on explicit multi-turn temporal metrics (e.g., time-to-flip / recovery curves), though agentic systems may involve iterative loops.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO includes explicit drift-vs-evidence controls and turn-by-turn trajectories (time-to-event, recovery-after-flip), that gives a clearer *temporal* and *causal* story than a single-shot judge label.
- If GALILEO reports robustness across diverse pressure operators (not just false-premise proof tasks), it supports broader generality.

## Where GALILEO is weaker / needs to improve

- BrokenMath’s benchmark design strongly stresses **“well-posed but false”** inputs + expert refinement; GALILEO should avoid ambiguous/ill-posed pressure conditions that can be dismissed as underspecification.
- The “difficulty → more sycophancy” finding suggests GALILEO should control for task difficulty (or at least stratify) to avoid confounding robustness with capability.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add/strengthen a benchmark-design principle: prefer **well-posed falsifications** over contradictions/underspecification; document quality checks.
- [ ] Stratify robustness metrics by **difficulty** (or proxy difficulty via baseline accuracy) and report interaction effects.
- [ ] Consider a categorical outcome taxonomy that distinguishes:
  - “accepts false premise and rationalizes it” vs
  - “flags inconsistency / refuses” vs
  - “corrects premise / reconstructs truth”,
  which can complement continuous drift metrics.
- [ ] If using LLM-judging anywhere, document judge model, prompt, and sensitivity checks, since BrokenMath relies on this evaluation mode.

## Quotes / details to potentially cite

- Abstract (motivation + benchmark): “LLMs … are prone to hallucination and sycophancy, often providing convincing but flawed proofs for incorrect mathematical statements provided by users.”
- Abstract (benchmark + size/design): “BrokenMath … built from advanced 2025 competition problems … perturbed with an LLM to produce false statements and subsequently refined through expert review.”
- Abstract (headline result): “the best model, GPT-5, producing sycophantic answers **29%** of the time.”
- Intro (limitations of prior math sycophancy benchmarks): focus on final-answer tasks; simple/contaminated datasets; ill-posed questions rather than well-posed falsehoods.
