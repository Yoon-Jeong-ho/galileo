# T3: Benchmarking Sycophancy and Skepticism in Causal Judgment

- Year: 2026
- Venue: arXiv
- Authors: Edward Y. Chang
- URL: https://arxiv.org/abs/2601.08258
- BibTeX key (if we add it): chang2026t3
- Tags: sycophancy, skepticism, causal-judgment, multi-turn, pressure, refusal, calibration, flip-metrics, inverse-scaling

## One-sentence takeaway

T3 is a diagnostic causal-judgment benchmark that decomposes performance into **Utility vs Safety** and measures **pressure-induced flips** and **calibrated refusal**, revealing both “skepticism traps” (over-refusal) and “sycophancy traps” (agreement under pressure).

## What problem does it solve?

- Standard causal-reasoning benchmarks often report aggregate accuracy, which confounds:
  - genuine reasoning errors,
  - safety-driven refusal/over-hedging,
  - and multi-turn instability under user pressure.
- The paper argues that for deployment-relevant robustness, we need **separate axes** for endorsing correct claims vs rejecting incorrect ones, plus explicit measurement of abstention/uncertainty behavior.

## What is the core method / protocol?

- **Dataset**: 454 expert-curated natural-language vignettes spanning **Pearl’s Ladder**:
  - L1 Association, L2 Intervention, L3 Counterfactual.
- **Labels**: three-way decision per item:
  - YES (valid), NO (invalid due to a causal trap), AMBIGUOUS (underdetermined).
  - Uses “Sheep” (YES) vs “Wolves” (NO) framing.
- **Trap taxonomy**: 12 recurring causal pitfalls (e.g., confounding, collider bias, Simpson’s paradox), across 10 domains.
- **Evaluation protocols** (three prompting conditions):
  1) Neutral Direct (baseline capability)
  2) Epistemic permissiveness (explicitly allows AMBIGUOUS; tests calibration)
  3) Adversarial Pressure (multi-turn robustness), including:
     - **Social pressure** (sycophancy): user pushes flawed logic
     - **Epistemic pressure** (self-doubt): “you’re wrong, rethink” interrogation
- **RCA (Recursive Causal Audit)**: a process-verified wrapper used as an *evaluation-only control condition* to test how operating points shift under stronger process constraints.

## What are the key metrics?

- **Utility (Sensitivity)**: true positive rate on Sheep (valid claims).
- **Safety (Specificity)**: true negative rate on Wolves (invalid claims).
- On underdetermined cases:
  - **Wise Refusal Rate (WRR)** (correct abstention)
  - **False Confidence Rate (FCR)** (overconfident non-abstention)
- Multi-turn stability under interrogation:
  - **Bad Flip Rate** = probability of changing a correct initial answer under pressure.
  - **Good Flip Rate** = probability of changing an incorrect initial answer under pressure.
  - Final accuracy after the pressure turn(s).

## What are the main results?

(From abstract/intro + results section as visible on the arXiv HTML.)

- **Skepticism trap at L1**: safety-tuned models can over-refuse; e.g., Claude Haiku rejects ~60% of valid associational claims (Utility ~40%) while maintaining high Safety.
- **Sycophancy trap at L2**: under pressure, models can flip correct rejections into endorsements; pressure sensitivity is not captured by neutral accuracy alone.
- **Scaling paradox at L3**: non-monotonic behavior where a “larger/newer” model underperforms an older one on ambiguous counterfactuals; described as paralysis/over-hedging rather than hallucination.
- RCA wrapper can shift the operating point (not claimed as core contribution), reducing over-hedging and restoring decisiveness in some cases.

## How is this similar to GALILEO?

- Very aligned with GALILEO’s focus on **multi-turn robustness under social/epistemic pressure**.
- Measures **instability dynamics** (flip rates), not just single-turn accuracy.
- Emphasizes separating “refusal/hedging” behavior from correctness—similar spirit to separating drift vs justified revision.

## How is this different from GALILEO?

- Domain is **causal judgment** with an explicit Pearl’s Ladder framing, rather than general belief-tracking / social-persuasion across arbitrary factual beliefs.
- Their decomposition is **Utility vs Safety (Sheep/Wolves)** and abstention metrics (WRR/FCR), whereas GALILEO is oriented around belief drift/revision controls + (likely) survival/time-to-failure and recovery.
- Uses a specific process wrapper (RCA) mainly as an evaluation control; GALILEO may instead emphasize interventions/recovery prompts as primary experimental factors.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO has explicit **drift-vs-evidence revision controls** and **recovery-after-flip** measurements, that’s a clearer mapping to real “misled then recover” scenarios than a static causal benchmark.
- If GALILEO uses time-to-event / survival-style metrics, it may provide more interpretable “when does failure occur?” summaries than flip rates alone.

## Where GALILEO is weaker / needs to improve

- GALILEO should consider adopting an explicit **sensitivity vs specificity** decomposition (or a close analogue), to avoid hiding “safe by refusing everything” and “helpful by agreeing with everything” regimes.
- Consider explicitly scoring **calibrated abstention** (WRR/FCR-like), to distinguish cautious correctness from over-hedging collapse.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a 2D reporting scheme analogous to **Utility vs Safety** for our tasks (e.g., endorse-true vs reject-false), in addition to any aggregate score.
- [ ] Track multi-turn **good-flip vs bad-flip** asymmetry as a robustness diagnostic alongside time-to-failure.
- [ ] Add a “wise refusal / calibrated uncertainty” bucket (and penalize over-refusal separately from wrong endorsement).
- [ ] In related work, cite T3 as evidence that (i) safety tuning can cause over-refusal, (ii) pressure can induce flips, and (iii) scaling can be non-monotonic due to hedging/paralysis.

## Quotes / details to potentially cite

- T3 decomposes performance into “Utility (sensitivity), Safety (specificity), and Wise Refusal on underdetermined cases.”
- Pressure protocols include “social pressure (sycophancy)” and “epistemic pressure (self-doubt)” variants that should not change the gold label.
- Flip metrics: Bad Flip Rate vs Good Flip Rate to separate indiscriminate reversal from selective correction.
