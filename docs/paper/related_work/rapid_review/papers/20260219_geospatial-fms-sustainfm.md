# Geospatial Foundation Models to Enable Progress on Sustainable Development Goals

- Year: 2025
- Venue: arXiv (HTML v1)
- Authors: Xiaokang Zhang, Weikang Yu, Aldino Rizaldy, Jian Wang, Chufeng Zhou, Richard Gloaguen, Gustau Camps-Valls
- URL: https://arxiv.org/html/2505.24528v1
- BibTeX key (if we add it): sustainfm2025
- Tags: geospatial, foundation-model, remote-sensing, benchmarking, SDG, sustainability

## One-sentence takeaway

SustainFM is an SDG-grounded benchmark (16 datasets/tasks) used to compare geospatial foundation models vs. “train-from-scratch” baselines, arguing evaluation should include transferability and energy/CO2—not just accuracy.

## What problem does it solve?

- The EO/FM literature has rapidly expanded, but it is unclear whether geospatial foundation models (FMs) actually help on societally-relevant, real-world tasks and how to evaluate them responsibly.
- Existing evaluations often emphasize task accuracy on narrow benchmarks, underweighting generalization/transfer and environmental cost.

## What is the core method / protocol?

- Introduce **SustainFM**, a benchmark aligned with the UN SDGs:
  - 16 tasks/datasets mapped to SDG 1–16 (SDG-17 is positioned as partnerships / framing), spanning multiple continents, sensors (e.g., Landsat, Sentinel-1/2, VIIRS, Gaofen-2, PlanetScope, Google Earth), and resolutions (~0.5m–30m).
  - Task types include regression, classification, semantic segmentation, and change detection.
- Evaluate a set of geospatial FMs (e.g., CROMA, RemoteCLIP, SoftCon, DOFA, GFM-Swin, Prithvi, SpectralGPT, ScaleMAE, SSL4EO-S12, SatlasNet) vs. baselines (ViT and ResNet-50 trained from scratch per task).
- Standardize downstream heads/decoders per task type:
  - Segmentation: UperNet; Change detection: Siamese UperNet; Classification/Regression: pooled features + FC.
- Fine-tuning strategy emphasized in results tables:
  - Freeze encoder, fine-tune decoder; additionally compare decoder-only vs full fine-tuning in an energy/CO2 analysis (CodeCarbon).

## What are the key metrics?

- Regression: RMSE (lower is better).
- Classification: mean F1 (mF1; higher is better).
- Efficiency/sustainability (in at least one focused comparison): training/inference energy (kWh) and CO2 emissions (kg) via **CodeCarbon**.
- Qualitative evaluation dimensions advocated: transferability, generalization under domain shift, robustness, energy efficiency.

## What are the main results?

- FMs are **often** better than training ViT/ResNet from scratch across diverse SDG tasks, but **not universally superior**.
- Decoder-only fine-tuning can be **much lower CO2** than full fine-tuning (paper reports relative CO2 increases for full FT vs decoder-only ranging roughly from ~+40% up to ~+168% depending on the model, on an SDG-15 flood-mapping task).
- Emphasizes a “beyond-accuracy” evaluation framing: impact-driven deployment, energy/CO2 reporting, and robustness/generalization.

## How is this similar to GALILEO?

- Shares the motivation of using EO/geospatial ML in service of real-world/social-good objectives rather than purely academic metrics.
- Reinforces that **generalization/transfer** across regions/tasks and **data efficiency** are central to geospatial ML value.

## How is this different from GALILEO?

- Primarily a **benchmarking + evaluation** contribution (SustainFM), not a new GALILEO-like method/model.
- Uses a relatively “standard” downstream protocol (frozen encoder + task decoder) rather than end-to-end or agentic/geospatial-reasoning pipelines (if GALILEO emphasizes those).
- Frames SDG alignment explicitly via mapping tasks to SDGs and discussing energy/ethics.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO targets unified multi-task training/inference or stronger cross-task transfer mechanisms, it may go beyond this paper’s largely per-task decoder fine-tuning setup.
- If GALILEO includes explicit evaluation for domain shift or real deployment constraints, it can operationalize their “impact-driven” recommendation.

## Where GALILEO is weaker / needs to improve

- GALILEO should ensure it reports **energy/CO2** (or proxy compute) and not only accuracy—this paper argues such reporting is part of responsible geospatial FM evaluation.
- If GALILEO’s evaluation suite is not SDG-grounded / societally-motivated, SustainFM suggests a more compelling framing and dataset/task diversity.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a brief “responsible evaluation” subsection: include energy/compute accounting and discuss transferability/generalization as first-class criteria.
- [ ] Consider adding (or citing) SustainFM as an SDG-grounded benchmark context; even if not used directly, use its taxonomy (tasks/sensors/resolutions) to justify evaluation breadth.
- [ ] If feasible, run at least one experiment that mimics their setup: freeze backbone, fine-tune small heads, and compare to training-from-scratch baseline to isolate the value of pretraining.

## Quotes / details to potentially cite

- “Evaluating FMs should go beyond accuracy to include transferability, generalization, and energy efficiency as key criteria…” (from abstract, paraphrase/near-quote).
- SustainFM summary: 16 datasets/tasks aligned with SDG1–16; global coverage (~200 regions), multi-sensor, 0.5–30m resolution.
- Reported claim: full fine-tuning vs decoder-only can increase CO2 by ~40% to ~168% depending on the FM (on their SDG-15 evaluation example).
