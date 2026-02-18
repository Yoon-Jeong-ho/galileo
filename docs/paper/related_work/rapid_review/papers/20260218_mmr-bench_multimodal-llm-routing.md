# MMR-Bench: A Comprehensive Benchmark for Multimodal LLM Routing

- Year: 2026
- Venue: arXiv
- Authors: Haoxuan Ma; Guannan Lai; Han-Jia Ye
- URL: https://arxiv.org/abs/2601.17814
- BibTeX key (if we add it): MMRBench2026Ma
- Tags: multimodal, routing, benchmark, cost-accuracy, budgets

## One-sentence takeaway

MMR-Bench proposes a standardized, budget-aware benchmark for *query-level routing among multiple MLLMs* and shows multimodal (image+text) routing signals can beat the best single-model accuracy at ~33% of its cost.

## What problem does it solve?

- In realistic deployments, no single multimodal LLM (MLLM) dominates across all tasks and difficulty levels; always using the strongest model wastes compute, while always using a cheap model loses accuracy.
- Existing routing benchmarks/methods are largely text-only; multimodal routing adds challenges (modality fusion, heterogeneous costs, no standard evaluation setup).

## What is the core method / protocol?

- Introduces **MMR-Bench**, a benchmark setup that isolates the multimodal routing problem:
  - fixed candidate model sets (“MLLM zoo”) with an explicit cost model,
  - inputs with modality availability (text, image; sometimes unimodal),
  - variable compute budgets,
  - tasks spanning OCR, general VQA, and multimodal math reasoning.
- Evaluates representative routing policies and references:
  - single-model baselines,
  - oracle upper bounds,
  - routers using **unimodal vs multimodal** signals (claim: multimodal cues reduce miscalibration on “text-in-image” / cluttered / low-res cases).

## What are the key metrics?

- **Cost–accuracy frontier** under fixed budgets / cost models (i.e., accuracy achieved at a given compute cost).
- Comparisons against strongest single model (accuracy at equal cost).
- Generalization of routing policies to unseen datasets / tasks (including transfer to text-only benchmarks).

## What are the main results?

- Multimodal routing signals improve routing quality vs text-only / image-only routing.
- Routed system can **match or exceed** the strongest single-model accuracy at roughly **~33% of the cost** (as stated in abstract/intro).
- Routing policies trained on a subset of models/tasks show **zero-shot generalization** to new datasets and even text-only benchmarks.

## How is this similar to GALILEO?

- Shared theme: **system-level evaluation** under constraints (budgets / heterogeneous conditions) rather than only improving a single base model.
- Useful precedent for building a *controlled benchmark harness* with:
  - explicit resource constraints,
  - strong baselines + oracle bounds,
  - “policy generalization” claims.

## How is this different from GALILEO?

- Focuses on **single-turn query routing** among MLLMs (cost vs accuracy), not multi-turn robustness issues (pressure, persuasion, sycophancy, drift, belief revision).
- Inputs are primarily multimodal tasks (OCR/VQA/math), not dialogue stability or social-adversarial settings.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO targets *multi-turn safety/robustness under pressure*, it addresses failure modes (drift, persuasion, sycophancy) that MMR-Bench does not operationalize.

## Where GALILEO is weaker / needs to improve

- If GALILEO makes claims about *adaptive deployment* (e.g., selecting policies/models across conditions), MMR-Bench is a concrete example of:
  - standardized cost modeling,
  - frontier-style reporting,
  - oracle upper bounds.

## Action items for GALILEO (experiments / method / writing)

- [ ] In related work, mention MMR-Bench as a **benchmarking template** for routing/adaptation under budgets; clarify how GALILEO’s “adaptation under pressure” differs (multi-turn + social robustness).
- [ ] Consider adding a small “resource trade-off” figure analogous to cost–accuracy frontier (if GALILEO has any knob like intervention strength / compute / number of rounds / deliberation budget).
- [ ] If GALILEO contemplates multi-model ensembles, cite MMR-Bench as motivation that **heterogeneous model zoos + routing** is a practical deployment pattern.

## Quotes / details to potentially cite

- “MMR-Bench … isolates the multimodal routing problem and enables comparison under fixed candidate sets and cost models.”
- “... routed system to exceed the strongest single model's accuracy at roughly 33% of its cost.”
- “Policies trained on a subset of models and tasks generalize zero-shot to new datasets and text-only benchmarks without retuning.”
