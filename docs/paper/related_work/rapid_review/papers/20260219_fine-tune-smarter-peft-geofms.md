# Fine-tune Smarter, Not Harder: Parameter-Efficient Fine-Tuning for Geospatial Foundation Models

- Year: 2025
- Venue: arXiv
- Authors: Benedikt Blumenstiel, Linus Scheibenreif, Paolo Fraccaro, Konrad Schindler
- URL: https://arxiv.org/abs/2504.17397
- BibTeX key (if we add it): Blumenstiel2025PEFTGeoFM
- Tags: peft, finetuning, geospatial, foundation-model, lora, vpt, adapters, segmentation

## One-sentence takeaway

Across several EO foundation models and datasets, LoRA / prompt-style PEFT can match (or beat) full fine-tuning while improving geographic generalization and cutting compute, with UNet-style decoders emerging as a strong default.

## What problem does it solve?

- Full fine-tuning of large geospatial foundation models (GeoFMs) is costly (GPU memory/time) and can degrade generalization via catastrophic forgetting.
- EO practitioners want *cheap adaptation* to downstream datasets, ideally with *better transfer to unseen geographic regions*.

## What is the core method / protocol?

- Systematic comparison of PEFT methods for GeoFMs:
  - **LoRA**: add low-rank adapters (r=16) to attention (Q,V) and MLP projections; train only LoRA weights.
  - **VPT-Deep**: add learnable prompt tokens (100 prompts) to *all* transformer layers.
  - **ViT Adapter** (only for Prithvi family due to engineering overhead): conv branch + cross-attention injections to add spatial inductive bias and multi-scale features.
- Evaluate PEFT across multiple pretrained EO backbones:
  - **DeCUR** (contrastive; per-modality ResNet-50; Sentinel-1+2)
  - **Prithvi 1.0** (MAE; ViT-B/16; HLS USA; 6 bands; short time series)
  - **Prithvi 2.0 300M** (MAE; larger ViT; global HLS; optional temporal/location metadata)
  - **Clay v1** (MAE + DINO-style distillation; ViT-B/8; multi-sensor; integrates metadata into positional embeddings)
- Decoder ablation (dense prediction):
  - Linear decoder (minimal, “probe-like”), FCN, UperNet, **UNet**.
- Experimental protocol:
  - For each (backbone × dataset × PEFT × decoder): Bayesian HPO over learning rate (16 trials), then 5 seeded runs; report averages.

## What are the key metrics?

- Primary: **mIoU** on segmentation tasks (explicitly referenced in Fig. 1).
- Generalization: **geographic hold-out sets (GHOS)** / held-out countries/regions (e.g., Bolivia subset for Sen1Floods11; Austria/Ireland hold-out for reBEN subset).
- Efficiency: training time and memory qualitative claims (and parameter-count overhead reported; VPT/LoRA up to ~2.4% extra params; ViT Adapter up to ~13.8%).

## What are the main results?

- **LoRA** often **matches or exceeds** full fine-tuning across datasets/backbones, while being much cheaper.
- PEFT can **improve generalization** to unseen geographic regions (GHOS) vs full fine-tuning.
- Architecture choices matter:
  - **UNet decoders** are recommended (better real-world dense prediction than linear probes).
  - **Fine-tuning without metadata** is suggested as the recommended configuration (at least for the settings explored; they explicitly note Prithvi 2.0 metadata is optional and evaluate “no metadata unless specified”).
- Practical contribution: implementations and configs integrated into **TerraTorch** (and code released).

## How is this similar to GALILEO?

- Same “GeoFM → downstream EO tasks” adaptation story.
- Strong emphasis on **robust evaluation across multiple datasets** and **generalization beyond the training geography**.
- Treats decoder design and protocol details (HPO, repeated seeds) as part of the scientific claim.

## How is this different from GALILEO?

- Focus is **adaptation strategy (PEFT)** rather than pretraining objective / data scaling / modality alignment.
- Restricts experiments to **multispectral optical** inputs for downstream evaluation (even though some backbones are multi-modal).
- Mostly a *systems+benchmarking* comparison paper; less about new representations.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO’s core contribution is in pretraining / representation learning, this paper is complementary rather than competing.
- If GALILEO supports broader modality/time handling, this paper’s scope (optical-only downstream eval) is narrower.

## Where GALILEO is weaker / needs to improve

- If GALILEO’s paper/stack assumes full fine-tuning, we may look compute-inefficient vs a PEFT baseline.
- If we do not report **geo hold-out** transfer, we risk under-claiming robustness vs this work.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add **LoRA** (and possibly VPT) as standard adaptation baselines for downstream tasks; report compute/memory deltas.
- [ ] Include at least one **geographic hold-out** evaluation slice (country/region hold-out) to support generalization claims.
- [ ] If we use metadata (lat/lon/time), add an ablation: **with vs without metadata** to check whether it truly helps.
- [ ] Use a “probe decoder” (linear) *and* a stronger decoder (e.g., UNet/UperNet) to separate representation quality from decoder capacity.

## Quotes / details to potentially cite

- “We demonstrate that PEFT techniques match or even exceed full fine-tuning performance and enhance model generalisation to unseen geographic regions, while reducing training time and memory requirements.” (abstract)
- “For LoRA, adapters with bottleneck dimension r=16 are added to … query and value matrices … and … feed-forward layers …” (Sec. 4)
- “For VPT, we include 100 learnable prompts in all transformer layers (VPT-Deep).” (Sec. 4)
- Datasets used: Sen1Floods11 (geo hold-out Bolivia), Burn Scars (new leakage-avoiding splits + 5km buffer), reBEN subset with GHOS (Austria/Ireland), m-Cashew, m-SA Crop Type (GEO-bench). (Sec. 4)
