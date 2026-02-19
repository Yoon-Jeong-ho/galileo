# Survey of Multimodal Geospatial Foundation Models: Techniques, Applications, and Challenges

- Year: 2025
- Venue: arXiv
- Authors: Liling Yang; Ning Chen; Jun Yue; Yidan Liu; Jiayi Ma; Pedram Ghamisi; Antonio Plaza; Leyuan Fang
- URL: https://arxiv.org/abs/2510.22964
- BibTeX key (if we add it): yang2025multimodal-gfm-survey
- Tags: geospatial;foundation-model;multimodal;survey

## One-sentence takeaway

A broad survey that taxonomizes *multimodal geospatial foundation models* by modality and evaluates representative models across downstream EO tasks, with a useful challenges/future-work section—but it is largely orthogonal to GALILEO’s focus (multi-turn social pressure / belief drift / sycophancy protocols).

## What problem does it solve?

- Consolidates and organizes a fast-growing literature on **multimodal geospatial/remote-sensing foundation models** (multi-resolution, multi-temporal, multi-sensor; plus vision-language).
- Identifies recurring challenges: modality heterogeneity, semantic gaps, distribution shift, efficiency/privacy/interpretability.

## What is the core method / protocol?

- Survey / taxonomy paper (no new model).
- Modality-driven organization covering “five core visual and vision-language modalities” (per abstract).
- Reviews techniques for alignment/integration/knowledge transfer, and adaptation paradigms.
- Includes an experimental comparison: “Representative multimodal visual and vision-language GFMs are evaluated across ten downstream tasks” (per abstract).

## What are the key metrics?

- Not specified on the abstract page; likely task-dependent remote-sensing metrics (e.g., mIoU/F1 for segmentation, mAP for detection, accuracy for classification, etc.).

## What are the main results?

- Provides a structured overview + empirical snapshot across 10 downstream tasks.
- Highlights open problems (generalization, interpretability, efficiency, privacy) and future directions.

## How is this similar to GALILEO?

- Shares the *general* “foundation models + evaluation” theme: broad capability claims and cross-task benchmarking.

## How is this different from GALILEO?

- Different application domain (earth observation / geospatial multimodal perception) vs GALILEO’s target (multi-turn conversational robustness under social pressure; belief drift / revision).
- Does not provide multi-turn protocols, pressure manipulations, flip/recovery metrics, or behavioral failure taxonomies related to sycophancy.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO is positioned as a behavioral evaluation protocol, it offers more controlled **causal tests** for social pressure / evidence vs. persuasion confounds than a survey-style EO evaluation overview.

## Where GALILEO is weaker / needs to improve

- Reminder: survey papers like this can be rhetorically strong; if reviewers expect a “landscape” section, we may want a similarly crisp taxonomy for *multi-turn drift / sycophancy / persuasion* work.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider borrowing the **survey structure** idea: a modality-/factor-driven taxonomy and a “challenges” section, but adapted to *pressure channels* (authority, consensus, politeness/face, repeated challenge, evidence injection, etc.).
- [ ] Likely **do not cite** unless GALILEO’s intro explicitly needs a general “foundation models in multimodal domains” analogy.

## Quotes / details to potentially cite

- Abstract (problem framing): “multimodal, multi-resolution, and multi-temporal characteristics of remote sensing data.”
- Abstract (scope): “covering five core visual and vision-language modalities.”
- Abstract (evaluation): “evaluated across ten downstream tasks.”
- Abstract (open problems): “domain generalization, interpretability, efficiency, and privacy.”
