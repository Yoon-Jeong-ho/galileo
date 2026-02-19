# Towards Knowledge Guided Pretraining Approaches for Multimodal Foundation Models: Applications in Remote Sensing

- Year: 2024
- Venue: arXiv
- Authors: Ajitesh Parthasarathy; Ankush Khandelwal; Rahul Ghosh; Vipin Kumar
- URL: https://arxiv.org/abs/2407.19660
- BibTeX key (if we add it): parthasarathy2024kgvsf
- Tags: remote-sensing; multimodal; self-supervised; pretraining; forecasting; causality

## One-sentence takeaway

Introduce a knowledge-guided, driver→response variable-step forecasting pretraining objective (weather→satellite imagery) that improves embeddings for downstream tasks where causal cross-modal dynamics matter.

## What problem does it solve?

- Standard SSL pretraining for geospatial foundation models (masked reconstruction; generic next-step prediction) tends to learn spatial structure and/or temporal smoothness, but does not explicitly encode *directional* cross-modal influence (e.g., weather drives vegetation/land-surface changes).
- Downstream tasks like crop type mapping, soil moisture estimation/forecasting, and missing/future image prediction benefit from representations that capture this driver→response relationship.

## What is the core method / protocol?

- Proposed pretraining task: **Knowledge Guided Variable-Step Forecasting (KG-VSF)**.
  - Treat forecasting as *conditional generation* of a **response modality** (Sentinel-2 spectral imagery) conditioned on **driver variables** (ERA5 weather) plus past imagery context.
  - Inputs: a short series of past images (irregularly available) + weather up to a chosen future date; target: image at that future date.
  - Variable horizon: last embedding forecasts a sampled future step (variable delta), similar in spirit to variable-lead-time forecasting.
- Architecture (high-level):
  - Image encoder: shared ViT over timestamps (patch embeddings, heavy encoder).
  - Weather encoder: unidirectional LSTM seq2seq producing per-day weather embeddings; subsampled/matched to image timestamps.
  - Time encodings: day-of-year embedding + delta-time embedding.
  - Multimodal sequence encoder: transformer with **forward-only (causal) attention** over time.
  - Forecaster: lightweight MLP/linear layers that combine current embedding + weather/time embeddings to produce forecast-time embedding; lightweight MLP decoder reconstructs image.
- Training protocol: **two-phase pretraining**
  - Phase 1: masked reconstruction pretraining of components (image MAE; weather masked reconstruction; sequence encoder reconstruction via masking embeddings).
  - Phase 2: forecasting with KG-VSF objective (uses weather as driver; compares against “no-weather” forecast baseline).

## What are the key metrics?

- Crop type mapping: average F1 score across crop classes (semantic segmentation).
- Soil moisture estimation/forecasting: regression metrics (reported as R^2 in tables for estimation; forecasting also evaluated).
- Image tasks (missing image prediction; future image forecasting): image reconstruction/forecast quality (paper discusses sharpness/blurriness qualitatively; likely MSE/PSNR-type metrics in full text).

## What are the main results?

- KG-VSF embeddings outperform embeddings from:
  - single-modality masked reconstruction (SM-MR),
  - multimodal masked reconstruction (MM-MR),
  - single-modality variable-step forecasting (SM-VSF),
  - multimodal variable-step forecasting (MM-VSF),
  on downstream tasks where driver-response dynamics matter.
- Notable pattern emphasized by authors: simply adding weather as an extra modality (MM-MR/MM-VSF) is not enough—**the pretraining objective needs to be directional/conditional** to capture the causal dependency.
- Forecasting-only pretraining (SM-VSF/MM-VSF) can yield blurrier imagery, whereas KG-VSF produces cleaner forecasts (per the authors’ description).

## How is this similar to GALILEO?

- Targets multimodal geospatial representation learning / foundation model pretraining.
- Uses self-supervised pretraining objectives and evaluates transfer to multiple downstream remote-sensing tasks.
- Emphasizes temporal modeling and forecasting-relevant representations (not just static image reconstruction).

## How is this different from GALILEO?

- Core novelty is **knowledge-guided driver→response conditional forecasting** (explicitly encoding directionality between modalities as the SSL task), rather than symmetric multimodal fusion or generic reconstruction.
- Uses a relatively explicit split of modalities: weather as “driver”, satellite imagery as “response”, and a causal/forward-only temporal encoder.
- Proposes a **stagewise** (two-phase) training recipe aimed at forcing causal cross-modal information into embeddings.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO already supports general multimodal tokenization across many sensors/modalities, it may be more extensible than a weather+Sentinel specific pipeline.
- If GALILEO’s architecture avoids modality-specific encoders (e.g., LSTM for weather), it may be architecturally cleaner/unified.

## Where GALILEO is weaker / needs to improve

- If GALILEO’s pretraining objectives are largely reconstruction/contrastive without a *directional* conditional forecasting task, it may miss gains on tasks requiring causal cross-modal dynamics.
- If GALILEO does forecasting, it may still be “symmetric” (predict all modalities) rather than explicitly modeling driver→response influence.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add/ablate a **driver→response conditional variable-horizon forecasting** pretraining objective (e.g., weather/context → future EO frame) and compare to masked reconstruction and symmetric multimodal forecasting.
- [ ] Try a **forward-only temporal attention** variant for pretraining to avoid leakage and align with forecasting.
- [ ] Add a write-up paragraph framing: “correlation-alignment (contrastive) vs causal/conditional generation objectives” and position GALILEO relative to KG-VSF.
- [ ] Downstream eval: include at least one task where weather→land-surface causality is strong (soil moisture estimation/forecasting; crop phenology mapping).

## Quotes / details to potentially cite

- “Existing pretraining approaches … do not fully capture the knowledge of causal interplay between different geospatial and environmental variables.”
- KG-VSF defined as: forecasting as conditional generation where “driver variables (e.g., weather) inform the prediction of response variables (e.g., satellite imagery).”
- Emphasis that causal direction is not encoded by multimodal fusion alone (MM-MR/MM-VSF), motivating knowledge-guided objective.
