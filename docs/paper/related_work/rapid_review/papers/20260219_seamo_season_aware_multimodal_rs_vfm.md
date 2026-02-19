# SeaMo: A Season-Aware Multimodal Foundation Model for Remote Sensing

- Year: 2024
- Venue: arXiv
- Authors: Chenyu Li; Gemine Vivone; Danfeng Hong
- URL: https://arxiv.org/abs/2412.19237
- BibTeX key (if we add it): seamo2024arxiv
- Tags: remote-sensing; multimodal; seasonal; masked-modeling

## One-sentence takeaway

SeaMo extends MAE-style masked image modeling to multi-season, multi-sensor (Sentinel-1 SAR + Sentinel-2 optical) remote-sensing pretraining via explicit temporal-multimodal fusion blocks and season-aware region selection.

## What problem does it solve?

- Remote-sensing foundation models often (a) focus on a single dimension (spatial *or* temporal *or* spectral), and/or (b) do not *explicitly* model seasonal variation when learning representations from time series satellite observations.
- Goal: learn a transferable VFM that uses the inherent seasonality/temporal continuity of EO data and fuses heterogeneous modalities.

## What is the core method / protocol?

- Base: masked autoencoder / masked image modeling (MIM) pretraining.
- Inputs: multi-season (T time points/seasons) multi-modal imagery: Sentinel-2 multispectral optical + Sentinel-1 SAR.
- Key design pieces (per paper text + figure captions):
  - **Unaligned / partially-overlapping spatial region selection** across seasons to increase spatial diversity and force robustness to seasonal change (regions at same time are selected identically across modalities; across different times, only partial overlap).
  - **Unified multimodal encoder**: concatenate visible patch tokens from optical+SAR for a season and run ViT self-attention to integrate across modality and space.
  - **Temporal-Multimodal (TM) fusion blocks**: cascade-style, cross-attention-based module to fuse information across seasons and modalities (explicit season-aware fusion rather than just stacking).
  - **Modality-specific decoders** to reconstruct masked content (MAE-style asymmetric encoder/decoder).
- Training strategy: **progressive / staged pretraining** moving from unimodal → multimodal → seasonal-multimodal learning.
- Pretraining data mentioned in the method section: **SSL4EO-S12** (Sentinel-2 12-channel + Sentinel-1 2-channel).

## What are the key metrics?

- Not extracted in detail from the accessible HTML snippet within the time budget; paper claims transfer to “a range of downstream geoscientific applications” with “extensive experiments and ablation studies”.
- When revisiting: record per-task metrics (likely classification mIoU/F1/accuracy; change detection; segmentation; etc.) and compare to SatMAE/CROMA/ScaleMAE baselines.

## What are the main results?

- Qualitative claim: better generalization/robustness/adaptability by explicitly modeling season-dependent attributes; “superior performance” reported across downstream tasks plus ablations.
- TODO for deeper pass: capture exact benchmark tables and the wins vs. (SatMAE, CROMA, ScaleMAE, SpectralGPT, etc.).

## How is this similar to GALILEO?

- Same general direction: leverage large-scale EO/RS pretraining to build transferable representations.
- Emphasizes multi-modal fusion (optical + SAR) and temporal structure, which is commonly relevant for EO foundation models.

## How is this different from GALILEO?

- SeaMo is explicitly framed as **season-aware** with architectural TM fusion blocks and explicit region-selection heuristics across seasons.
- SeaMo is squarely **MIM/MAE-style reconstruction** driven, whereas GALILEO may prioritize different objectives/architectures (depending on the current GALILEO design).

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO already has a simpler, unified temporal modeling story (e.g., one clean temporal transformer without staged training), it may be easier to explain/implement.

## Where GALILEO is weaker / needs to improve

- If GALILEO does not explicitly model seasonality (or does not test season-shift robustness), SeaMo suggests this can be a meaningful gap.
- If GALILEO’s multimodal fusion is late-fusion or naive stacking, SeaMo’s unified encoder + TM block is a concrete alternative.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a “seasonality/temporal shift robustness” evaluation slice (train on subset of seasons, test on held-out seasons; or measure performance drop across season splits).
- [ ] Consider a **partial-overlap region selection** scheme across time as a data/patch sampling ablation.
- [ ] Consider an explicit **cross-attention temporal fusion block** (or similar) for multi-season fusion, and ablate against simpler pooling/concatenation.
- [ ] If writing related work: position against SatMAE/CROMA/Skysense as “explicit season-aware multimodal MIM”.

## Quotes / details to potentially cite

- Abstract (motivation + method keywords): “SeaMo leverages a masked image modeling framework to fully exploit the spatial, spectral, and seasonal dimensions of RS data.”
- Figure-caption-level method summary: “images from the same temporal instance are selected identically across various modalities, while images from different instances exhibit partial overlaps… The TM block effectively merges features from multiple seasons and modalities.”
- Pretraining dataset detail (method section): “pretrained on the SSL4EO-S12 dataset… 12-channel multispectral optical imagery from Sentinel-2 and 2-channel SAR backscatter data from Sentinel-1.”
