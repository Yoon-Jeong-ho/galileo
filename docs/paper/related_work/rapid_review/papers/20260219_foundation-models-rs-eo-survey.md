# Foundation Models for Remote Sensing and Earth Observation: A Survey

- Year: 2024
- Venue: arXiv (survey)
- Authors: Aoran Xiao; Weihao Xuan; Junjue Wang; Jiaxing Huang; Dacheng Tao; Shijian Lu; Naoto Yokoya
- URL: https://arxiv.org/html/2410.16602v3
- BibTeX key (if we add it): xiao2024foundation
- Tags: remote-sensing, earth-observation, foundation-model, survey, multimodal

## One-sentence takeaway

A comprehensive (and timely) survey/taxonomy of remote-sensing foundation models (VFMs/VLMs/LLM-adjacent/generative), emphasizing why general-domain FMs underperform on RS (especially non-optical) and summarizing datasets, training paradigms, and benchmarking.

## What problem does it solve?

- Provides a structured overview of the rapidly growing RS foundation model (RSFM) space, where methods, datasets, modalities, and evaluation practices are fragmented.
- Clarifies key domain-specific obstacles for transferring general FMs to RS/EO: modality mismatch (MSI/HSI/SAR/LiDAR/DSM/TIR), different viewpoints/statistics, resolution/scale variability, temporal dynamics, and limited web-scale RS pretraining corpora.

## What is the core method / protocol?

- Survey + taxonomy + benchmarking narrative:
  - Defines RSFM and organizes work into (at least) VFMs for RS, VLMs for RS, and other RSFMs (LLMs, generative models, etc.).
  - Reviews pretraining paradigms relevant to RS: supervised pretraining (often ImageNet vs RS-specific), self-supervised learning (contrastive; masked image modeling / MAE-style; hybrids), and adaptation (full finetune vs PEFT vs zero-shot).
  - Covers modality-specific considerations and tasks (scene classification, segmentation, detection incl. oriented boxes, change detection, RS VQA/captioning/grounding).

## What are the key metrics?

- As a survey, it is not defined by a single metric; evaluation is discussed across common RS tasks/datasets (classification/segmentation/detection/change detection, etc.).
- The paper emphasizes transfer/generalization (few-shot / zero-shot), and robustness across sensors/resolutions/modalities as the practical "metrics" of interest for RSFMs.

## What are the main results?

- Key conclusions/messages (qualitative):
  - General-domain FMs work best on optical RGB-like RS imagery, but performance degrades sharply for non-optical and/or high-dimensional modalities (MSI/HSI/SAR/LiDAR), motivating RS-specific pretraining and architectures.
  - RS-specific pretraining datasets and SSL objectives are central; however, RS pretraining datasets remain much smaller/less diverse than natural-domain corpora.
  - Highlights emerging directions: multimodal pretraining (paired modalities aligned by geolocation), temporal modeling for multi-time EO, and parameter-efficient adaptation as a practical strategy.

## How is this similar to GALILEO?

- If GALILEO is positioned as a foundation-model-like system for EO/RS, this survey provides:
  - A shared framing (pretrain-at-scale then adapt) and the standard set of downstream EO tasks.
  - A vocabulary/taxonomy to describe GALILEO relative to VFMs/VLMs/multimodal RSFMs and (crucially) relative to sensor modalities.

## How is this different from GALILEO?

- This work is a survey/benchmarking overview, not a new model or training recipe.
- Broad coverage across modalities/tasks; likely shallower than GALILEO-specific design details/experiments.

## Where GALILEO is stronger / cleaner (if true)

- A focused method paper can provide:
  - Clear, reproducible training setup and ablations (the survey necessarily aggregates heterogeneous results).
  - A single unified benchmark/eval protocol (if GALILEO provides one).

## Where GALILEO is weaker / needs to improve

- If GALILEO currently targets only a subset of modalities (e.g., optical), this survey underscores pressure to:
  - Extend to non-optical modalities and/or paired multimodal training (e.g., MSI+SAR, DSM, LiDAR).
  - Explicitly address resolution/scale variability and temporal dynamics.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add 1-2 related-work paragraphs using this survey as the umbrella citation for "RS foundation models" and for the RS-vs-natural domain-gap argument.
- [ ] Use the paper's taxonomy (VFM/VLM/other RSFMs; modality list) to structure GALILEO related work and positioning.
- [ ] Ensure the GALILEO evaluation story explicitly mentions (and, if possible, tests) robustness across sensors/resolutions and (if relevant) temporal generalization.
- [ ] If GALILEO uses PEFT, cite the survey's discussion framing PEFT as especially relevant for RS due to domain gaps.

## Quotes / details to potentially cite

- Definition-style framing of foundation models: "any model trained on broad data (typically using self-supervision at scale) that can be adapted ... to a wide range of downstream tasks" (cited in their intro).
- Their explicit claim that general-domain FMs can show "degraded performance and even failures" on RS data of various non-optical modalities, motivating RSFMs.
- The modality list (RGB/MSI/HSI/SAR/LiDAR/TIR/DSM) as a compact justification of why EO is fundamentally multimodal and out-of-distribution vs natural images.
