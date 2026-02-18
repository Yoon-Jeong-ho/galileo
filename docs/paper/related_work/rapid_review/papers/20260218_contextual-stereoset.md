# Contextual StereoSet: Stress-Testing Bias Alignment Robustness in Large Language Models

- Year: 2026
- Venue: arXiv
- Authors: Abhinaba Basu
- URL: https://arxiv.org/abs/2601.10460
- BibTeX key (if we add it): basu2026contextualstereoset
- Tags: bias-eval, context-sensitivity, robustness, stereotyping, benchmarking

## One-sentence takeaway

Bias/"stereotype selection" scores can swing substantially under *non-adversarial* contextual reframings (time/place/audience/style), so evaluations should report **context sensitivity fingerprints** rather than a single fixed-condition bias number.

## What problem does it solve?

- Standard bias benchmarks (e.g., StereoSet) typically test a fixed prompt framing and report a scalar score, implicitly assuming it generalizes to deployment.
- The paper argues that *routine context cues* ("in 1990", "in India", "as gossip", "to someone unlike you") can change measured stereotype selection without changing the underlying stereotype content.
- Goal: provide a **stress test of evaluation robustness** (not a claim about real-world base rates of bias).

## What is the core method / protocol?

- **Contextual StereoSet** extends StereoSet by holding the 3-way options fixed (stereotype / anti-stereotype / unrelated) while varying a contextual *premise* along a factorial grid:
  - location (country), year, style (direct vs gossip; plus an intermediate "talk" in the full grid), observer similarity ("someone like you" vs "someone unlike you").
- Forced-choice instruction is framed as typicality: select the option "most people would consider typical" (answers decoded back to S/A/U with stored option-order randomization).
- Two tracks:
  - **Full-grid diagnostic**: 12 locations × 5 years × 3 styles × 2 observers = 360 contexts per item (run on a smaller item set).
  - **Budgeted protocol (exp2)**: reduced grid (72 contexts) + baselines; scaled to 4,229 items for screening.
- Reporting artifact: **Context Sensitivity Fingerprints (CSF)**
  - per-dimension dispersion (stddev across levels) and paired contrasts (e.g., 1990−2030; gossip−direct; dissimilar−similar)
  - bootstrap CIs + permutation tests; BH-FDR correction.

## What are the key metrics?

- **SS (stereotype selection rate)**: fraction of contexts where the stereotypical option is selected.
- **Dispersion** across context dimensions (mean per-item stddev across levels): e.g., σ_year, σ_loc, σ_style, σ_obs.
- **Paired contrasts**: ΔSS for targeted comparisons (1990 vs 2030; gossip vs direct; dissimilar vs similar), with uncertainty + significance.

## What are the main results?

- Across 13 models / two protocols, **context changes shift measured stereotype selection** without adversarial prompting.
- Examples highlighted in the abstract/introduction:
  - anchoring to **1990 vs 2030** increases stereotype selection (reported as significant in tested contrasts),
  - **gossip framing** increases stereotype selection in most full-grid models,
  - **out-group observer** framing can shift rates by up to ~13 percentage points.
- Effects also show up in exploratory high-stakes **vignettes** (hiring, lending, help-seeking), suggesting sensitivity is not confined to benchmark probes.

## How is this similar to GALILEO?

- Same meta-point: **single-number evaluations can be misleading** because model behavior depends on *interaction conditions*.
- CSF is conceptually similar to our desire for an evaluation artifact that captures **where models fail** rather than only aggregate success.
- Provides a concrete example of a *robustness-to-natural-variation* framing (no adversarial search required) that we can reuse rhetorically.

## How is this different from GALILEO?

- Focuses on **bias/stereotype selection** under contextual reframing, not multi-turn **belief drift / pressure-driven flips / recovery**.
- Largely **single-shot forced-choice** probes (multi-context but not multi-turn dynamics).
- Constructs measured are typicality judgments ("most people") rather than epistemic truthfulness under evidence vs pressure.

## Where GALILEO is stronger / cleaner (if true)

- GALILEO can (and should) separate **pressure-only drift** from **evidence-driven belief revision**, which this benchmark does not target.
- Our multi-turn trajectory metrics (time-to-flip, recovery, oscillation) address *temporal dynamics* missing here.

## Where GALILEO is weaker / needs to improve

- We may be under-emphasizing **contextual fragility under benign framing** (time/place/audience/style) as a first-class axis.
- We lack a compact "fingerprint" summary analogous to CSF that stakeholders can use for regression testing.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a "**context framing robustness**" subsection: test whether our drift/flip rates change when prompts include benign anchors (e.g., year=1990 vs 2030; "gossip" vs "direct" style; in-group vs out-group audience).
- [ ] Consider a **fingerprint-style report**: per-operator dispersion + paired contrasts (pressure operator × context anchor), with bootstrap CIs.
- [ ] In writing, borrow the rhetorical reframing: ask "**under what conditions does failure appear?**" rather than "is the model robust?".

## Quotes / details to potentially cite

- "A model that avoids stereotypes in a lab benchmark may not avoid them in deployment. We show that measured bias shifts dramatically when prompts mention different places, times, or audiences—no adversarial prompting required."
- Contribution framing: Contextual StereoSet + two tracks (360-context diagnostic grid; budgeted 4,229-item protocol) and the CSF summary (dispersion + contrasts with bootstrap CIs and FDR correction).
