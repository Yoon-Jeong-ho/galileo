# Prescriptive Scaling Reveals the Evolution of Language Model Capabilities

- Year: 2026
- Venue: arXiv
- Authors: Jikai Jin; Vasilis Syrgkanis; Sham Kakade
- URL: https://arxiv.org/abs/2602.15327
- BibTeX key (if we add it): jin2026prescriptive
- Tags: scaling, capability-eval, compute-to-performance, quantile-regression, observational-evals

## One-sentence takeaway

They propose “prescriptive scaling” via estimating high-quantile, monotone-saturating capability frontiers (benchmark score vs log pretraining FLOPs) from large observational leaderboard data, and show most task frontiers are temporally stable except math reasoning.

## What problem does it solve?

- Practitioners want a *prescriptive* mapping from a pretraining compute budget to an *attainable* downstream benchmark score after contemporary post-training (rather than only average scaling trends).
- They also want to know whether such compute→capability mappings remain reliable as training/post-training recipes evolve over time.

## What is the core method / protocol?

- Collect large-scale observational evaluation data for many post-trained models, keyed by base-model pretraining compute (FLOPs).
- For each benchmark/task, estimate a **capability boundary** defined as a **high conditional quantile** (e.g., ~0.98 quantile) of performance as a function of **log(FLOPs)**.
- Fit the boundary using **smoothed quantile regression** with a **monotone, saturating sigmoid parameterization** (intended to model ceilings / saturation).
- Validate **temporal reliability** by fitting on earlier model generations and evaluating on later releases (chronological splits).
- Additional analyses:
  - task-dependent saturation behavior (which tasks hit stable ceilings vs evolve)
  - contamination-related shifts for math reasoning (they mention AIME-2025; report no clear evidence of inflation)
  - an adaptive sampling algorithm to recover near-full-data frontiers with reduced eval budget.

## What are the key metrics?

- Benchmark scores/accuracies (leaderboard-style tasks; paper references Open LLM Leaderboard suites and additional frontier leaderboards).
- Pretraining compute expressed as FLOPs (modeled on log-scale).
- Fit quality / predictiveness of the estimated high-quantile frontier (implicitly via held-out time validation).
- Evaluation cost for the adaptive sampling method (fraction of total evaluation budget needed to recover the frontier).

## What are the main results?

- Capability boundaries for many tasks are **mostly stable over time**, suggesting compute is a fairly deterministic predictor of *attainable* post-trained performance envelopes.
- **Math reasoning** is an exception: its boundary appears to **consistently advance** across time.
- They release a dataset (**Proteus-2k**, described as ~2.4k open-weight models evaluated post Open-LLM-Leaderboard-v2 cutoff) and combine it with ~5k existing observations.
- Adaptive sampling can recover near-full-data frontiers using roughly **~20%** of the evaluation budget (and sometimes less, depending on task).

## How is this similar to GALILEO?

- Treats *evaluation over time* as a first-class object and asks whether performance envelopes shift as the ecosystem changes.
- Focuses on methodology for turning messy, heterogeneous evaluation corpora into a reliable, decision-oriented summary (frontier/boundary) rather than point estimates.

## How is this different from GALILEO?

- This work is primarily **observational/retrospective**: it mines large public evaluation repositories and fits statistical frontiers.
- The target output is a **compute→attainable-score map** (high-quantile boundary), not necessarily a new evaluation benchmark design or mechanistic explanation.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO emphasizes controlled protocols, dataset/benchmark construction, or causal/diagnostic evaluation: GALILEO can offer cleaner attribution than observational frontier fitting.
- If GALILEO has explicit safeguards around leakage/contamination and measurement validity: can complement their coarse contamination case study.

## Where GALILEO is weaker / needs to improve

- If GALILEO lacks a simple, decision-facing summary like “attainable frontier at compute budget,” adopting a boundary/quantile framing could improve practicality.
- If GALILEO does not currently validate stability across time/model generations, their chronological-split validation is a good template.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding a “capability boundary” view: estimate high-quantile performance frontiers (vs mean curves) for key tasks/metrics.
- [ ] Add a temporal validation protocol: train/fit on earlier model families/releases, test on later ones; explicitly report which tasks’ frontiers drift.
- [ ] If we build/curate eval suites, quantify saturation/ceiling behavior and track ceiling drift over time (especially for math-like tasks).
- [ ] Consider whether an adaptive evaluation scheduling/sampling algorithm could reduce eval cost while preserving frontier estimates.

## Quotes / details to potentially cite

- They define *prescriptive scaling* as mapping a compute budget to a targeted downstream performance level after post-training.
- “Capability boundaries—high conditional quantiles of benchmark scores as a function of log pre-training FLOPs—via smoothed quantile regression with a monotone, saturating sigmoid parameterization.”
- Temporal finding: boundaries “mostly stable,” except “math reasoning … consistently advancing boundary over time.”
- Efficiency claim: recover near-full-data frontiers with roughly “20% of evaluation budget.”
