# TiMo: Spatiotemporal Foundation Model for Satellite Image Time Series

- Year: 2025
- Venue: arXiv
- Authors: Xiaolei Qin, Di Wang, Jing Zhang, Fengxiang Wang, Xin Su, Bo Du, Liangpei Zhang
- URL: https://arxiv.org/html/2505.08723v1
- BibTeX key (if we add it): timo2025
- Tags: remote-sensing, satellite-image-time-series, spatiotemporal, foundation-model, masked-image-modeling, hierarchical-transformer, attention

## One-sentence takeaway

TiMo is a hierarchical ViT foundation model for satellite image time series that replaces early-stage full self-attention with a structured spatiotemporal attention (STGA/D-STGA) and is pretrained with masked modeling on a 1M-sample, 10-timestamp Sentinel-2 dataset (MillionST), yielding strong gains across several SITS tasks.

## What problem does it solve?

- Existing SITS foundation models often adapt MAE/VideoMAE with *plain* ViTs that flatten space×time tokens and run global attention, which can miss multiscale spatiotemporal structure of aligned SITS and be data-limited by small timestamp counts.
- Goal: learn general spatiotemporal representations for many downstream SITS tasks with limited labels.

## What is the core method / protocol?

- Backbone: 4-stage *hierarchical* vision transformer (Swin-like) with convolutional patch embedding + downsampling to produce multi-scale pyramid features.
- Key module: **Spatiotemporal Gyroscope Attention (STGA)**
  - For each token at (x,y,t), attention is restricted to tokens that share:
    - the same time t (spatial context within a frame), OR
    - the same spatial location (x,y) across other times (temporal context for that pixel/patch).
  - This explicitly leverages spatial alignment in SITS.
- Efficiency variant: **Differential STGA (D-STGA)**
  - Approximates spatial similarity over time using a median temporal feature and per-time differences, reducing complexity from ~O(T^2) to ~O(T) (as claimed).
- Pretraining: MAE-like masked image modeling adapted for hierarchical architectures (they cite Hiera-style pretraining); during pretraining they keep standard MHSA in all blocks, then *swap in* STGA/D-STGA in early stages for downstream fine-tuning.
- Data: **MillionST** (introduced here)
  - 100k locations, 10 timestamps (2017–2022 at ~6-month intervals), ~1M Sentinel-2 samples; sampling concentrated around populous cities (Europe/N. Africa/W. Asia) using a Gaussian around city centers.

## What are the key metrics?

- Deforestation segmentation: mIoU (MultiEarth)
- Multi-class land-cover segmentation: segmentation accuracy metrics (reported in their tables; exact numbers not captured in this rapid read)
- Crop type classification: accuracy (MTLCC)
- Flood detection: classification accuracy (Sen12Flood) and segmentation mIoU (KuroSiWo)

## What are the main results?

- Across multiple tasks (deforestation monitoring, land-cover segmentation, crop classification, flood detection), TiMo variants outperform or match prior SITS foundation models they compare against (SeCo, CACo, GASSL, SatMAE, Prithvi-EO-2.0, etc.).
- They also show that pretraining SatMAE on MillionST improves it, arguing the richer timestamp count matters.

## How is this similar to GALILEO?

- Same broad theme: learn transferable representations for Earth observation / remote sensing via large-scale pretraining and evaluate on diverse downstream tasks.
- Emphasis on spatiotemporal modeling and leveraging time series structure, not just single images.

## How is this different from GALILEO?

- Focuses specifically on **aligned optical SITS** (Sentinel-2) and builds *architectural inductive bias* directly into attention (restricted “gyroscope” neighborhoods).
- Pretraining objective is MAE-like masked modeling adapted to hierarchical networks; downstream swaps attention modules post-pretrain.
- Introduces/leans heavily on their new dataset MillionST with fixed 10 timestamps over 5 years (not general multimodal or broader EO modalities).

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO aims for broader modality/task coverage (e.g., multimodal EO, different sensors, geography), TiMo’s MillionST sampling (cities-heavy; particular regions) may be narrower and may raise generalization questions.
- TiMo’s method relies on good spatial alignment; approaches that tolerate misalignment or irregular sampling might generalize better.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently uses vanilla global attention over flattened space×time, TiMo suggests a concrete inductive bias (separable space-at-t + time-at-(x,y)) that could improve sample efficiency and interpretability.
- Efficiency: D-STGA proposes a path to reduce temporal attention cost.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider an ablation/variant that restricts attention to (same t) ∪ (same (x,y) across time) as a lightweight inductive bias baseline.
- [ ] Add a discussion paragraph in related work: hierarchical backbones + structured spatiotemporal attention for SITS foundation models (TiMo as a representative).
- [ ] If relevant, compare against TiMo-style pretraining/fine-tuning mismatch (MHSA pretrain → STGA finetune) as a design pattern.

## Quotes / details to potentially cite

- Abstract (problem framing): existing SITS FMs “encode entire temporal sequences without explicitly capturing multiscale spatiotemporal relationships between land objects”.
- Core idea: STGA attends only to tokens sharing the same temporal position or the same spatial position across time; “gyroscope” refers to the geometric structure of attention regions.
- Dataset: MillionST contains “1 million Sentinel-2 images … 100,000 geographic locations … 10 temporal phases over five years.”
