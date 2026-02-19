# Uncovering Competency Gaps in Large Language Models and Their Benchmarks

- Year: 2025
- Venue: arXiv
- Authors: Matyas Bohacek; Nino Scherrer; Nicholas Dufour; Thomas Leung; Christoph Bregler; Stephanie C. Y. Chan
- URL: https://arxiv.org/abs/2512.20638
- BibTeX key (if we add it): competencygaps2025bohacek
- Tags: evaluation, benchmarks, robustness, interpretability, SAE, competency-gaps

## One-sentence takeaway

Use sparse autoencoder (SAE) concept features to decompose benchmark performance into per-concept “coverage” (benchmark gaps) and per-concept “accuracy” (model gaps), revealing systematic weaknesses that aggregated scores hide.

## What problem does it solve?

- Standard benchmark scores are aggregated and can mask (a) *model gaps* (specific concept areas where a model underperforms) and (b) *benchmark gaps* (concept areas underrepresented in the benchmark itself).
- Existing topic labels / clusters are often coarse or manual and don’t scale to fine-grained coverage auditing.

## What is the core method / protocol?

- Define a *concept dictionary* using a pre-trained sparse autoencoder (SAE) over internal model activations.
- For each benchmark item, compute:
  - SAE concept activations (“which concepts are present/active for this item”).
  - A per-item performance signal (correct/incorrect or score).
  - A *saliency weighting* so concepts more causally/diagnostically involved in the item can be emphasized.
- Aggregate across benchmark items to produce:
  - **Benchmark coverage** per concept: which SAE concepts the benchmark exercises a lot vs. barely at all.
  - **Model performance** per concept: which concepts are systematically associated with low performance.
- Can compare across multiple benchmarks by placing them in the shared SAE concept space.
- Paper frames this as a method called **Competency Gaps (CG)**.

## What are the key metrics?

- Per-concept benchmark coverage score (paper denotes cross-benchmark coverage as χ_bench(c)).
- Per-concept model score / performance score (cross-benchmark χ_model(c)).
- (From the arXiv HTML) The paper emphasizes “saliency-weighted performance scores” rather than raw frequency-weighted accuracy.

## What are the main results?

- Demonstration on two open-source instruction-tuned models (Gemma2-2B-Instruct; Llama 3.1-8B-Instruct) across 10 benchmarks.
- **Model gaps:** underperformance on concepts that *counter sycophancy* (e.g., politely refusing, asserting boundaries) and concepts connected to *safety discussions*.
- **Benchmark gaps:** many benchmarks over-represent concepts around *obedience/authority/instruction-following* and under-represent concepts that should be within their intended scope.
- The method is positioned as complementary to aggregate benchmark metrics: it explains “why a model scored as it did”.

## How is this similar to GALILEO?

- Shares the goal of moving from coarse aggregate eval to *diagnostic, actionable breakdowns*.
- Suggests an analysis workflow: run evaluation, identify conceptual gaps, then augment data/benchmarks—similar to an iterative eval→fix loop.

## How is this different from GALILEO?

- CG is representation/interpretability-grounded: concepts come from SAEs over internal activations, not from external task taxonomies.
- It is primarily an *analysis layer* over existing benchmarks, not a new training or inference-time control method.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO’s target is reliable system behavior (e.g., safety/robustness) under real interaction, GALILEO may provide more direct end-to-end protocols than concept-space auditing.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently reports mostly aggregate metrics, this paper is a strong reminder that we may be blind to “competency gaps” that only appear after slicing by latent concepts.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding a “concept coverage / concept failure” section in eval reporting (even if not SAE-based initially): systematically identify *where* failures cluster.
- [ ] If we can access SAEs for our base model(s), prototype a lightweight CG-style analysis on our eval sets to see whether failures align with specific concept clusters.
- [ ] In related work: cite CG as an approach for *representation-grounded benchmark auditing* and for explaining aggregate score variance.

## Quotes / details to potentially cite

- “Aggregated metrics can obscure (i) particular sub-areas where the LLMs are weak (‘model gaps’) and (ii) imbalanced coverage in the benchmarks themselves (‘benchmark gaps’).” (abstract)
- “We propose a new method that uses sparse autoencoders (SAEs) to automatically uncover both types of gaps… computing saliency-weighted performance scores…” (abstract)
- Finding: models “underperformed on concepts that stand in contrast to sycophantic behaviors … and concepts connected to safety discussions.” (abstract)
- Finding: benchmarks “over-represented concepts related to obedience, authority, or instruction-following…” (abstract)
