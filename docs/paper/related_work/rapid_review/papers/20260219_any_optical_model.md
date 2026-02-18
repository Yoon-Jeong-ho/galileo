# Any-Optical-Model: A Universal Foundation Model for Optical Remote Sensing

- Year: 2025
- Venue: AAAI 2026 (arXiv preprint)
- Authors: Danfeng Hong et al. (see arXiv)
- URL: https://arxiv.org/abs/2512.17224
- BibTeX key (if we add it): anyopticalmodel2025hong
- Tags: earth-observation, optical, remote-sensing-foundation-model, multispectral, multi-sensor, multi-resolution, masked-autoencoder

## One-sentence takeaway

AOM proposes an optical remote-sensing foundation model that can ingest arbitrary spectral band sets and spatial resolutions via per-band tokenization plus resolution-adaptive patch embedding, and reports strong robustness for missing-band, cross-sensor, and cross-resolution transfer.

## What problem does it solve?

- Existing remote-sensing foundation models (RSFMs) are typically pretrained assuming a fixed set of bands (e.g., specific Sentinel-2 channels) and a fixed spatial resolution / patch scale.
- In practice, optical sensors differ in band layouts (missing/new bands) and ground sampling distance; this hurts deployment for:
  - missing-band scenarios
  - cross-sensor transfer / fusion (e.g., Sentinel-2 vs Landsat vs HLS)
  - unseen resolutions (sub-meter to 100m-scale imagery)

## What is the core method / protocol?

- **Spectrum-independent Tokenizer (SiTok)**
  - Processes each channel (band) individually with a shared patch-embedding conv, producing per-band token grids.
  - Adds an explicit **band embedding / channel index encoding** (sinusoidal on channel index) so spectral identity is preserved even when bands are missing or newly introduced.
  - Concatenates tokens across channels to feed a Transformer.
- **Multi-scale Adaptive Patch Embedding (MAPE)**
  - Uses a **bank of convolutional kernels** with different receptive fields (different patch sizes).
  - Selects the closest kernel to a target patch size and (if needed) applies **pseudo-inverse resize** (per FlexiViT-style PI-resize) to adapt weights, aiming to be more stable than large-factor resizing.
  - Intended to adapt spatial granularity across wide resolution ranges.
- **Pretraining objective**
  - Channel-wise self-supervised masking + reconstruction (MAE-like) to learn spectral-spatial structure.
  - Adds a **multi-scale semantic alignment** constraint/objective to encourage consistent global semantics across scale embeddings.

## What are the key metrics?

- Paper claims SOTA across >10 public datasets/benchmarks; specific metrics depend on each downstream task.
- Focused robustness settings:
  - performance under **band missing**
  - **cross-sensor** transfer
  - **cross-resolution** transfer

## What are the main results?

- Reports consistent SOTA performance under the above challenging settings on datasets including Sentinel-2, Landsat, and HLS, and evaluation on Geo-Bench (per paper).
- The headline is robustness/generalization rather than a single-task in-domain gain.

## How is this similar to GALILEO?

- Same high-level goal: **general-purpose geospatial/EO foundation modeling** that transfers across dataset/domain shifts.
- Explicitly targets **sensor heterogeneity** and **scale variability**, which are core practical issues for EO.
- Uses self-supervised pretraining objectives (masked modeling) to learn broadly reusable representations.

## How is this different from GALILEO?

- AOM is framed as an **optical RSFM** focused on multispectral optical channels and resolution changes; it is not presented as a general multi-modal geospatial model beyond optical.
- Architecture emphasizes **per-band tokenization + band identity embeddings** and **patch-embedding kernel adaptation**; depending on GALILEO’s design, GALILEO may not rely on explicit per-band tokenization.
- Pretraining includes a specific **multi-scale alignment** mechanism tied to multi-scale patch embeddings.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO supports broader modalities (beyond optical multispectral) or more unified treatment of time/metadata, it may be positioned as more general than AOM’s optical focus.
- If GALILEO avoids hand-designed per-channel encodings, it may be simpler to extend across sensors with different band semantics (but this depends on GALILEO’s current tokenizer).

## Where GALILEO is weaker / needs to improve

- If GALILEO currently assumes fixed band sets or fixed spatial resolution during pretraining/fine-tuning, AOM is a direct warning sign: real deployments need explicit handling for arbitrary band configurations and resolution shifts.
- Consider whether GALILEO has an explicit mechanism for:
  - missing-band robustness
  - cross-sensor transfer without retraining
  - stable multi-scale semantics

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a small evaluation section (or appendix) explicitly testing **missing-band** and **cross-sensor** settings (e.g., train/pretrain on one sensor, evaluate on another) if not already present.
- [ ] Add **cross-resolution** stress tests: evaluate at resolutions not seen in pretraining, or with synthetic resampling.
- [ ] Consider a tokenizer baseline inspired by SiTok: per-band embeddings + explicit band identity encoding; test whether it improves robustness.
- [ ] If GALILEO already has multi-scale pathways, consider adding an explicit **multi-scale semantic alignment** loss/regularizer and ablate.

## Quotes / details to potentially cite

- Motivation: optical sensors have diverse band layouts and GSD; fixed-band/fixed-resolution RSFMs break under missing bands, cross-sensor fusion, and unseen scales.
- Method summary (from abstract): “spectrum-independent tokenizer” with dedicated band embeddings; “multi-scale adaptive patch embedding”; “multi-scale semantic alignment”; “channel-wise self-supervised masking and reconstruction.”
