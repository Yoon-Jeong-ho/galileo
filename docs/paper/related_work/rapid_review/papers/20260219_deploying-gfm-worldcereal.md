# Deploying Geospatial Foundation Models in the Real World: Lessons from WorldCereal

- Year: 2025
- Venue: arXiv (mentions ICML / Machine Learning in header; treat as preprint)
- Authors: Kristof Van Tricht; Gabriel Tseng; Giorgia Milli; David Rolnick; Ruben Cartuyvels; Inbal Becker Reshef; Zoltan Szantoi; Hannah Kerner
- URL: https://arxiv.org/html/2508.00858
- BibTeX key (if we add it): vantricht2025worldcereal
- Tags: geospatial, foundation-model, deployment, protocol, crop-mapping, presto, evaluation

## One-sentence takeaway

A practical, three-step protocol (requirements → adaptation → empirical testing) for operationalizing geospatial foundation models, demonstrated by integrating and evaluating Presto inside the WorldCereal global crop-mapping pipeline.

## What problem does it solve?

- Benchmark-centric evaluations for geospatial foundation models often miss deployment-critical factors (data heterogeneity, resource constraints, integration with legacy pipelines, qualitative map artifacts), so practitioners lack a “recipe” for moving from promising papers to an operational mapping system.

## What is the core method / protocol?

- Proposes a structured deployment protocol:
  - Step 1: Define application requirements + hypotheses (e.g., generalization needs, CPU-only constraints, input covariates, evaluation criteria including qualitative map checks).
  - Step 2: Choose an adaptation strategy (e.g., finetune vs freeze; optional intermediate SSL to adapt to preprocessing / distribution shift).
  - Step 3: Empirically test using splits that reflect operational realities (geographic holdouts, temporal holdouts, plus random split), plus qualitative artifact inspection.
- Case study: WorldCereal crop mapping (10m, end-of-season global maps; allow end users to retrain models without GPUs) using Presto (pixel-level multi-source time series: Sentinel-1/2, DEM, weather).

## What are the key metrics?

- Primary: F1 scores (per-class F1 and/or macro F1), evaluated under:
  - Random split (in-distribution)
  - Geographic split (held-out countries)
  - Temporal split (held-out year)
- Also emphasized: qualitative map quality (visual inspection of predicted patches; artifacts/tiling issues).

## What are the main results?

- In both tasks (binary cropland and multiclass crop type), fine-tuned *pretrained* Presto outperforms supervised CatBoost baselines and a randomly initialized Presto architecture.
- Strong geographic + temporal generalization is demonstrated via held-out countries / years.
- The additional SSL adaptation step (intended to address preprocessing shifts and label scarcity) did **not** provide meaningful improvements over straightforward supervised finetuning in their experiments.
- Practical note: highlights compute footprint differences via MAC comparisons for 12-timestep time series (example: Presto 38.37M MACs vs Galileo-Nano 89.40M vs AnySat 889.94M).

## How is this similar to GALILEO?

- Treats geospatial FM value as primarily: transfer + generalization under distribution shift (space/time) and label scarcity.
- Emphasizes evaluation beyond random splits, including geographic and temporal generalization—core to how GALILEO should be positioned and tested.
- Explicitly considers operational constraints (CPU-only retraining, latency/scale), which aligns with “deployability” narratives around smaller GALILEO variants.

## How is this different from GALILEO?

- This is primarily a *deployment protocol + case study* paper, not a new foundation model architecture/training recipe.
- The model in focus is Presto and a specific operational stack (WorldCereal + openEO pipeline + specific covariates/timeseries formatting).
- GALILEO may target broader modalities and pretraining objectives; this paper’s novelty is mostly in evaluation methodology and deployment framing.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO has a unified pretraining story across tasks/modalities, it can contribute more on the “why representations work” side; this paper is intentionally practitioner-oriented.
- If GALILEO provides clearer scaling curves / ablations, that would complement this protocol-oriented work.

## Where GALILEO is weaker / needs to improve

- This paper is a reminder that we need “operational splits” (geo/temporal + qualitative map checks) as first-class artifacts; if GALILEO evaluations are mostly benchmark/rand-split, it will look less deployment-ready.
- Compute footprint is explicitly compared (incl. Galileo-Nano); we should be prepared to justify MAC/latency vs accuracy tradeoffs and provide CPU-friendly deployment guidance.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a “deployment protocol” subsection (requirements → adaptation → testing) to related work / framing, citing this paper as motivation for evaluation beyond benchmarks.
- [ ] Ensure GALILEO paper reports at least one *geographic* and one *temporal* generalization experiment (not only random split), plus at least one qualitative artifact analysis for dense map outputs.
- [ ] Include a compact compute table (MACs/params/latency) and explicitly position Galileo-* variants for CPU-bound retraining/inference settings.
- [ ] Consider an ablation showing that (when finetuning) additional SSL-style adaptation is not always beneficial—mirroring their finding—so readers have a decision rule.

## Quotes / details to potentially cite

- “Standardized evaluation tasks often fail to capture real-world complexities relevant for end-user adoption such as data heterogeneity, resource constraints, and application-specific requirements.” (Abstract)
- Proposed protocol: “defining application requirements, adapting the model to domain-specific data and conducting rigorous empirical testing.” (Abstract)
- Compute comparison example (12-timestep time series): “MAC operations ranging from 38.37M for Presto … to 89.40M for Galileo-Nano … to 889.94M for AnySat …” (Section 2.1, compute resources bullet)
