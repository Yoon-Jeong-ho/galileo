# PyViT-FUSE: A Foundation Model for Multi-Sensor Earth Observation Data

- Year: 2025
- Venue: ICLR 2025 ML4RS Workshop
- Authors: Manuel Weber, Carly Beneke
- URL: https://arxiv.org/abs/2504.18770 (HTML: https://arxiv.org/html/2504.18770v1)
- BibTeX key (if we add it): Weber2025PyViTFuse
- Tags: earth-observation, foundation-model, multi-sensor, fusion, multi-resolution, self-supervised

## One-sentence takeaway

A self-supervised EO foundation model that fuses an arbitrary set of mixed-resolution sensor bands via attention into patch tokens processed by a pyramidal ViT, trained with a SwAV-style objective (no pixel reconstruction).

## What problem does it solve?

- Existing vision foundation models assume fixed 3-channel RGB and a single spatial resolution; EO often has many bands across sensors with different native resolutions.
- Many EO self-supervised approaches rely on masked reconstruction (MAE-like), which can be brittle for EO imagery; they want band-combination-agnostic embeddings without a decoder.

## What is the core method / protocol?

- **Inputs:** co-registered multi-modal EO “area of view” patches; each modality may include multiple bands at different resolutions.
- **Per-band patchification at native resolution** (no resampling), followed by **learnable linear projections** into a shared feature dimension.
- **Fusion module:** multi-head cross-attention that uses a learned query to compute attention weights over bands and forms a **weighted sum (fused patch token)**; designed to accept an *arbitrary number* of input bands.
- **Backbone:** a **pyramidal Vision Transformer** stack operating on the fused patch-token sequence.
- **Pretraining objective:** adapts **SwAV** (cluster assignment swapping between augmented views) for self-supervised learning, avoiding a reconstruction decoder and encouraging embeddings that are less tied to specific band subsets.
- **Interpretability hook:** visualize fusion attention scores to see which bands/sensors are used.

## What are the key metrics?

- Not clearly specified in the arXiv abstract/landing page; downstream evaluation appears to include at least one **segmentation** setting (paper mentions an appendix segmentation example).
- (TODO if needed later) Extract exact datasets/metrics from PDF (e.g., mIoU / F1 / accuracy / retrieval).

## What are the main results?

- Demonstrates **applicability to downstream tasks** and **interpretability** of the band-fusion mechanism via attention visualizations.
- Key claimed benefit: embeddings can be **robust to different input band combinations**, enabling reuse across tasks/sensors (e.g., Sentinel-2 alone vs Sentinel-1+2 to mitigate clouds) with the same encoder.
- Quantitative numbers were not present in the arXiv abstract; needs PDF skim to capture headline scores.

## How is this similar to GALILEO?

- Targets **EO foundation modeling** and **sensor/modal fusion** rather than a single fixed input format.
- Emphasis on **self-supervised pretraining** and producing reusable representations for multiple downstream tasks.

## How is this different from GALILEO?

- Uses a **SwAV-style clustering objective** (contrastive/clustering) rather than masked reconstruction-style objectives.
- Explicitly frames the “arbitrary band set / mixed resolution” challenge and solves it with **attention-based band fusion** + pyramidal ViT.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO uses simpler, more standardized input pipelines (fixed sensor suite / consistent resolution), training and evaluation may be easier to reproduce and compare.
- If GALILEO has stronger benchmarking across tasks/datasets, it may present clearer evidence than a workshop paper.

## Where GALILEO is weaker / needs to improve

- If GALILEO must support **variable sensor availability** and **mixed resolutions**, PyViT-FUSE’s “native-res band tokenization + attention fusion” is a concrete blueprint.
- If GALILEO relies heavily on MAE-style reconstruction, this paper is a reminder that **non-reconstruction SSL** (SwAV-like) can be attractive for EO.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider an ablation/idea section discussing **band-agnostic encoders**: train with random band subsets and evaluate robustness to missing modalities.
- [ ] Explore **attention-based fusion over bands** (learned query cross-attn) as an alternative to early concat or fixed pooling, especially when band count changes.
- [ ] If writing related work: cite as an example of **mixed-resolution, multi-sensor EO foundation models** trained with **SwAV-style SSL**.

## Quotes / details to potentially cite

- Abstract: “fuse an arbitrary number of mixed-resolution input bands into a single representation through an attention mechanism.”
- Abstract: “train the model … in a self-supervised manner, leveraging core concepts of the SwAV algorithm.”
- Intro motivation: MAE pixel-space reconstruction can be challenging in EO; SwAV avoids the need for a decoder and aims for embeddings independent of band combinations.
