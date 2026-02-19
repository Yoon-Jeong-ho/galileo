# How to Train Your Long-Context Visual Document Model

- Year: 2026
- Venue: arXiv
- Authors: Austin Veselka
- URL: https://arxiv.org/abs/2602.15257
- BibTeX key (if we add it): Veselka2026TrainLongContextVisualDoc
- Tags: long-context, vlm, document-vqa, training-recipes, synthetic-data, preference-optimization

## One-sentence takeaway

A large-scale, fairly reproducible recipe/ablation study for pushing VLMs to very long document contexts (up to ~344K), highlighting simple high-impact tricks (page indices, match train/eval length) plus a practical comparison of CPT vs SFT vs LongPO and evidence of vision↔text long-context transfer.

## What problem does it solve?

- Open-weight long-context VLMs for **long PDF / document VQA** have recently gotten strong (e.g., Qwen3-VL, GLM-4.5/6V), but their **training recipes + data pipelines** are not well specified.
- The paper aims to provide actionable, end-to-end guidance for how to train long-context visual document models and what actually matters.

## What is the core method / protocol?

- Multi-stage training study across:
  - **Continued pretraining (CPT)** with scalable/self-supervised tasks adapted to long visual documents (e.g., visual FIM, unshuffle, key/position retrieval, counting).
  - **Supervised finetuning (SFT)** on synthetic long-document VQA (question generation + answer generation via distillation).
  - **Preference optimization** via **LongPO** (short-to-long preference optimization, derived from DPO).
  - **Self-improvement**: use their synthetic pipelines such that the (weaker) model can generate/bootstraps data.
- Practical training engineering:
  - Two stages (short vs long) to reduce overhead when targeting very high sequence lengths.
  - **Model merging** to mitigate catastrophic forgetting / preserve base instruct behavior.
- Synthetic answer generation variants:
  - “Plain distillation” (pass full example to strong teacher) vs a **recursive pipeline** that extracts evidence per-page, ranks pages, and conditions the teacher on the most relevant subset.

## What are the key metrics?

- Aggregate metrics designed to reduce benchmark-specific noise:
  - **Visual-LC Avg (VA)**: average (normalized) across MMLongBenchDoc, MMLBD-C, MMLongBench, DUDE, SlideVQA.
  - **LC Avg (LCA)**: VA + text long-context benchmarks **HELMET** and **LongBench v2**.
- Primary tie-breaker: **MMLBD-C** (their corrected MMLongBenchDoc variant).

## What are the main results?

- **Training length matching**: Training on context lengths similar to evaluation benchmarks can outperform training on longer contexts (reported ~1.4–3.0 VA improvement).
- **Page indices**: Prepending explicit page indices during both training and evaluation yields a strong boost (reported +2.8 on MMLBD-C and +2.8 on visual-LC average).
- **CPT vs SFT**:
  - If compute is constrained and base context length is already sufficient, **SFT alone can be competitive** on document-VQA-centric metrics, but CPT can help on text LC (HELMET).
- **LongPO**:
  - LongPO can improve VA more than SFT (paper reports +2.1 VA vs SFT in one comparison) but costs >2x compute.
  - For best MMLBD-C specifically, plain distillation SFT can be more effective.
- **Vision→text transfer**:
  - They report that training on visual long-context tasks can substantially improve **long-context text** performance (not just the usual text→vision transfer).
- Benchmark hygiene:
  - Release **MMLBD-C**: corrected/filtered variant of MMLongBenchDoc (modify 251/1091, remove 16).

## How is this similar to GALILEO?

- If GALILEO involves any **long-context** setting (multi-page inputs; long sequences; retrieval-within-context), this paper’s takeaways are directly relevant:
  - match train/eval length distribution rather than “always train longer”,
  - explicit **index/position tokens** can be a cheap, high-leverage feature,
  - beware catastrophic forgetting and consider merging/replay-like mitigations.

## How is this different from GALILEO?

- The paper is centered on **visual document VQA** (PDF pages / images) and scaling context length in VLMs; it is not primarily about remote sensing or geospatial foundation modeling (if that is GALILEO’s focus).
- Many results depend on specific long-document benchmarks (MMLongBenchDoc family) and the synthetic data pipelines targeting that format.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO’s core claims are outside document VQA (e.g., geospatial, temporal modeling, multi-sensor fusion), then GALILEO’s domain focus and evaluation may be cleaner/more targeted than this broad LC VLM study.

## Where GALILEO is weaker / needs to improve

- If GALILEO uses long sequences at all, this paper suggests GALILEO may be leaving performance on the table without:
  - explicit **position/page indices**,
  - careful **length distribution** alignment between training and evaluation,
  - safeguards against **forgetting** when extending context or adding LC tuning.

## Action items for GALILEO (experiments / method / writing)

- [ ] If we have any long-context evaluation, add a length-matching ablation: train on (A) longer-than-eval vs (B) matched-to-eval distributions.
- [ ] Add an “explicit index tokens” ablation (page index / timestamp index / tile index; whatever the analog is in GALILEO).
- [ ] Consider a “recursive evidence selection” baseline (extract evidence per segment then answer) to separate reasoning vs brute-force full-context distillation.
- [ ] If we do any LC finetuning, add a quick forgetting check + consider merging/replay.

## Quotes / details to potentially cite

- “Training on context lengths that match evaluation context lengths outperforms training on longer contexts.”
- “Training and evaluating with page indices provides a simple, high-impact boost to long-document performance.”
- They define **Visual-LC Avg (VA)** and **LC Avg (LCA)** as normalized aggregates across multiple LC benchmarks.
