# Earth Observation Foundation Model PhilEO: Pretraining on the MajorTOM and FastTOM Datasets

- Year: 2025
- Venue: arXiv
- Authors: Nikolaos Dionelis, Riccardo Musto, Jente Bosmans, Simone Sarti, Giancarlo Paoletti, Peter Naylor, Valerio Marsocci, Sébastien Lefèvre, Bertrand Le Saux, Nicolas Longépé
- URL: https://arxiv.org/html/2506.14765v2
- BibTeX key (if we add it): 
- Tags: earth-observation, foundation-model, self-supervised-learning, scaling-laws, compute, benchmarking

## One-sentence takeaway

PhilEO scales an EO foundation model to much larger pretraining data (FastTOM 2TB / MajorTOM 23TB) and model sizes (up to 200M), comparing U-Net vs ViT vs Mamba on PhilEO Bench and emphasizing linear-complexity alternatives to attention.

## What problem does it solve?

- How to scale Earth Observation (EO) foundation model pretraining to very large unlabeled datasets (tens of TB) while keeping training/inference computationally feasible.
- Empirically: what happens to downstream performance (roads, buildings, land cover) when scaling dataset size, model size, and architecture.

## What is the core method / protocol?

- Self-supervised pretraining for EO imagery (Sentinel-2) with PhilEO-style pretext tasks (paper references prior PhilEO work: masked reconstruction + geo-location estimation as pretext tasks).
- Scaling axes:
  - Data scaling: PhilEO Globe (~0.5TB, land-only) -> FastTOM (2TB, land-only subset) -> MajorTOM Core-S2L2A (23TB, land+ocean+ice).
  - Model scaling: ~44M params and up to ~200M params.
  - Architecture scaling: Geo-aware U-Net CNNs, ViT backbones (with UPerNet decoder), and 2D Mamba SSM models (linear complexity; bidirectional for images).
- Evaluation: fine-tune and compare on PhilEO Bench downstream tasks:
  - road density estimation (regression)
  - building density regression
  - land cover semantic segmentation
- Additional: FLOPs profiling to compare computational cost (highlighting quadratic attention vs linear Mamba/U-Net scaling).

## What are the key metrics?

- Downstream task metrics:
  - RMSE for road density and building density regression.
  - Accuracy (and/or segmentation accuracy) for land cover mapping.
- Compute/efficiency:
  - FLOPs as an architecture-agnostic compute proxy.

## What are the main results?

- Larger pretraining data and larger models can improve downstream results on PhilEO Bench (e.g., improvements reported for road density regression when moving from smaller to larger pretraining sets).
- ViT has quadratic complexity in token length due to attention; Mamba is presented as linear-complexity and thus more scalable for large-area/high-resolution imagery.
- Geo-Aware U-Net variants and larger (e.g., ~200M) models tend to be among the strongest performers across many n-shot settings; Mamba variants can be comparable while using less compute.
- The paper also reports comparisons against other EO GFMs (e.g., TerraMind, Prithvi-EO-2.0) in the context of dataset properties and/or benchmarking.

## How is this similar to GALILEO?

- High-level only: it is an evaluation-forward paper that emphasizes controlled comparisons across model variants and scaling dimensions.

## How is this different from GALILEO?

- Different domain and problem: EO foundation model scaling and pixel-level downstream tasks vs GALILEO’s multi-turn belief-consistency / persuasion / drift-and-recovery evaluation of LLMs.
- Different evaluation object: vision/remote-sensing encoders and segmentation/regression heads vs dialog models under multi-turn conversational pressure.

## Where GALILEO is stronger / cleaner (if true)

- GALILEO targets a clearly specified conversational failure mode (multi-turn drift under pressure) with explicit control arms (neutral re-asking) and turn-of-failure/recovery dynamics; this paper is not about multi-turn dialog dynamics.

## Where GALILEO is weaker / needs to improve

- Not directly applicable; however, GALILEO’s paper could borrow the "scaling axes" framing (data/model/architecture) as a writing pattern when discussing robustness scaling (e.g., more rounds, stronger personas, larger models).

## Action items for GALILEO (experiments / method / writing)

- [ ] (Writing pattern) Consider explicitly presenting GALILEO scaling axes (e.g., rounds, persona strength, model size, prompting/guardrails) analogous to data/model/architecture scaling in other ML domains.

## Quotes / details to potentially cite

- MajorTOM Core-S2L2A pretraining subset size: 23TB (land, oceans, ice).
- FastTOM subset size: 2TB (land-only; excludes oceans and ice).
- Complexity claim: ViT attention has quadratic complexity in token sequence length; Mamba SSM models achieve linear complexity and may scale better to large-area EO imagery.
