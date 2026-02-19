# A Genealogy of Foundation Models in Remote Sensing

- Year: 2025 (arXiv; accepted Jan 2026)
- Venue: ACM SIGSPATIAL (accepted; arXiv preprint)
- Authors: Kevin Lane (et al.; see arXiv for full author list)
- URL: https://arxiv.org/abs/2504.17177
- BibTeX key (if we add it): lane2025genealogy
- Tags: remote-sensing; foundation-model; self-supervised-learning; survey; multi-sensor; multimodal

## One-sentence takeaway

A survey/genealogy of remote-sensing foundation models that organizes RS SSL objectives (contrastive variants + masked image modeling) and highlights RS-specific adaptations and open problems, especially around multi-sensor / multi-modal Earth observation.

## What problem does it solve?

- Remote-sensing foundation models are proliferating, but many are direct ports of computer-vision SSL recipes; this paper clarifies the design space and RS-specific pitfalls/opportunities (spectral bands, time/seasonality, multi-sensor).
- Fills a gap in prior RS-FM surveys by focusing less on broad benchmarking tables and more on *how* SSL objectives are adapted (e.g., temporal positives, multi-scale views, indices as reconstruction targets, multi-sensor pairing).

## What is the core method / protocol?

- Survey + taxonomy (“genealogy”) approach:
  - Frames RS SSL foundation models as mostly inheriting from CV SSL families:
    - contrastive learning with negative sampling (e.g., MoCo-style)
    - distillation-based contrastive learning (e.g., BYOL/DINO-style)
    - redundancy reduction (e.g., Barlow Twins/VICReg-style)
    - masked image modeling (MAE-style)
  - Starts with single-sensor models, then emphasizes multi-sensor / multi-modal EO as the key RS-specific axis.
- Discusses RS data characteristics that stress naive CV transfers (multiple spectral bands, different sensor physics like SAR vs optical, scale distributions, seasonality/temporal variation).

## What are the key metrics?

- As a survey, it mainly references common downstream-eval regimes used across RS foundation-model papers (task transfer performance after pretraining), rather than introducing a new metric.
- The paper’s evaluation emphasis is conceptual: representation quality and strategies to reduce compute needs (“democratize” training via better SSL/recipes).

## What are the main results?

- Consolidates the RS foundation-model landscape into an objective-based taxonomy and argues that:
  - RS should be treated as a distinct modality (not just “images”), due to spectrum + sensor physics + spatiotemporal structure.
  - Multi-sensor training (e.g., SAR+optical and other modalities) is under-exploited relative to the availability of paired EO observations.
  - There are clear opportunities for leveraging unlabeled seasonal/time-series EO at scale.

## How is this similar to GALILEO?

- If GALILEO is positioned as a “foundation-model-like” approach, this paper is useful as:
  - a template for writing a genealogy/taxonomy section (objective families, data modes, single-sensor → multi-sensor progression)
  - motivation language around unlabeled data scale, transfer, and compute democratization

## How is this different from GALILEO?

- This work is remote-sensing-specific and primarily a survey; it does not propose a new training objective, benchmark, or intervention.
- If GALILEO’s core contribution is *behavioral robustness / multi-turn interaction / social-pressure evaluation* (as suggested by current TOP10.md), then this paper is off-theme and should likely not be cited except as an example of “genealogy-style related work” structure.

## Where GALILEO is stronger / cleaner (if true)

- GALILEO (as currently framed in our shortlist) appears to have a sharper, testable experimental protocol + metrics; this paper is descriptive rather than protocol-defining.

## Where GALILEO is weaker / needs to improve

- If GALILEO involves multimodality or sensor fusion: this paper is a reminder that *multi-sensor* structure (paired modalities, time/seasonality) needs explicit handling and should be described cleanly (what is a “modality”, what alignments exist, what invariances are assumed).

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider borrowing the “genealogy” organization for the related-work section (roots → families → RS-specific adaptations / or, analogously, roots → families → GALILEO-specific adaptations).
- [ ] If relevant, add a short paragraph clarifying what counts as a modality/sensor/time-slice in GALILEO and what invariances we assume.

## Quotes / details to potentially cite

- Categorization: RS SSL foundation models “fall broadly into one of four categories”: (1) contrastive via negative sampling, (2) contrastive via distillation, (3) contrastive via redundancy reduction, (4) masked image modeling. (Intro)
- Emphasis: multi-sensor aspect of Earth observations and opportunities leveraging “unlabeled, seasonal, and multi-sensor remote sensing observations.” (Abstract)
