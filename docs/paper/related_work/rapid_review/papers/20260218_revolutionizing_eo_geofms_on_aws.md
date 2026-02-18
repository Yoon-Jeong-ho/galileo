# Revolutionizing earth observation with geospatial foundation models on AWS

- Year: 2025
- Venue: AWS Machine Learning Blog (technical how-to)
- Authors: Karsten Schroer; Bishesh Adhikari; Iza Moise
- URL: https://aws.amazon.com/blogs/machine-learning/revolutionizing-earth-observation-with-geospatial-foundation-models-on-aws/
- BibTeX key (if we add it): awsblog2025_geofm_aws
- Tags: geospatial foundation models, ViT, MAE pretraining, Clay, embeddings, similarity search, change detection, SageMaker, vector database

## One-sentence takeaway

A practitioner-oriented recipe for using a geospatial foundation model (Clay) as a frozen encoder to produce 768-d embeddings for large-scale similarity search and embedding-distance change detection, with an end-to-end AWS/SageMaker pipeline.

## What problem does it solve?

- How to operationalize GeoFMs for Earth observation analytics at scale (batch embedding generation, indexing, and downstream tasks) without training a bespoke model from scratch.
- Concrete patterns for two common EO workflows:
  - image-to-image semantic similarity search over large areas
  - time-series change detection via embedding drift.

## What is the core method / protocol?

- Treat the GeoFM as an **encoder-only ViT** that maps a preprocessed satellite chip to an embedding.
- Data prep:
  - Retrieve Sentinel-2 imagery from AWS Open Data (S3).
  - Chip into **256×256** patches (for Sentinel-2, ~2.56 km × 2.56 km at 10 m).
  - Filter clouds; normalize bands.
- Embedding generation:
  - Run Clay v1 on GPU instances; output **768-d** vectors (CLS and/or patch embeddings).
- Downstream:
  - Similarity search: cosine similarity + ANN index in a vector DB (OpenSearch Serverless or LanceDB).
  - Change detection: compute distances over time; visualize via PCA; optionally do harmonic regression vs a baseline seasonality model and threshold deviations.
- Optional fine-tuning pattern:
  - Freeze encoder; train a lightweight task head (MLP for classification; decoder for segmentation).

## What are the key metrics?

- Similarity search quality is illustrated qualitatively (no standard retrieval metric reported in the post).
- Change detection uses deviation from a fitted baseline (harmonic regression) and thresholding (again mainly qualitative).
- For their illustrative segmentation fine-tune, they report:
  - Validation IoU and F1 (numbers below).

## What are the main results?

- Demonstrates an end-to-end, scalable pipeline on SageMaker Pipelines for: chipping → embeddings → post-processing → indexing.
- Reports a fast segmentation fine-tuning example (Impact Observatory LULC):
  - After 1 epoch: **85.7% validation IoU**
  - After 10 epochs: **92.4% IoU**, **95.6% F1**
- Provides a concrete deployment repository: https://github.com/aws-samples/sample-geospatial-foundation-models-on-aws

## How is this similar to GALILEO?

- Uses the same broad idea of **foundation-model representations for EO** as a reusable substrate.
- Emphasizes **embedding-based** workflows (retrieval/similarity; change detection) that GALILEO likely also benefits from (e.g., representation learning + downstream tasks).
- Highlights the “frozen encoder + small head” fine-tuning pattern (useful baseline for GALILEO comparisons).

## How is this different from GALILEO?

- This is an **engineering/deployment** post, not a new modeling contribution; focuses on AWS architecture and pipelines.
- Centers on **Clay v1** and a specific set of choices (256×256 chips, 768-d embeddings, cosine similarity, OpenSearch/LanceDB).
- Change detection approach is largely **heuristic/unsupervised** (embedding distance + baseline regression) rather than a learned change detector.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO provides a more principled training objective, benchmarking, and ablations, it will be stronger scientifically than this blog’s qualitative demonstrations.
- If GALILEO supports richer spatiotemporal modeling or multi-sensor fusion beyond the blog’s default Sentinel-2 pipeline, that’s a differentiator.

## Where GALILEO is weaker / needs to improve

- Reproducible, “push-button” **systems story**: this post is a good template for a clear, modular, production-friendly pipeline narrative.
- Retrieval + vector DB integration narrative: if GALILEO uses embeddings, this suggests concretely documenting ANN indexing, schema, and operational considerations.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a short “Operationalization / Scaling” subsection (even if not AWS-specific) describing: chipping, batch embedding generation, ANN indexing, and retrieval latency/throughput.
- [ ] If GALILEO claims change detection utility, benchmark **embedding-distance drift baselines** (cosine distance + simple thresholding; PCA + baseline regression) as cheap, strong baselines.
- [ ] Consider a small “frozen encoder + light head” baseline (segmentation/classification) to contextualize any heavier fine-tuning.

## Quotes / details to potentially cite

- GeoFMs “excel as embedding models for geospatial similarity search and ecosystem change detection” and can be fine-tuned “with minimal labeled data” for downstream tasks.
- The blog frames three use cases: “geospatial similarity search”, “embedding-based change detection”, and “custom geospatial machine learning (fine-tuning)”.
- Concrete implementation details worth citing in prose (not as a claim of novelty): 256×256 chips; 768-d embeddings; OpenSearch Serverless or LanceDB for ANN; harmonic regression baseline for seasonal time series.
