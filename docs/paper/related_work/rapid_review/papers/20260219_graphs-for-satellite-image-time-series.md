# On the use of Graphs for Satellite Image Time Series

- Year: 2025
- Venue: arXiv (submitted to IEEE; preprint)
- Authors: Corentin Dufourg; Charlotte Pelletier; Stéphane May; Sébastien Lefèvre
- URL: https://arxiv.org/html/2505.16685v1
- BibTeX key (if we add it): dufourg2025graphsSITS
- Tags: remote-sensing, SITS, spatiotemporal-graphs, OBIA, GNNs, survey

## One-sentence takeaway

A survey + end-to-end pipeline for turning satellite image time series (SITS) into **spatio-temporal graphs** (object-based nodes + spatial/temporal edges) and applying graph analytics/GNNs to classification/regression tasks, with case studies in land-cover mapping and water forecasting.

## What problem does it solve?

- Pixel-grid SITS are large and complex (spatial × temporal × spectral), making modeling and computation challenging.
- Pure OBIA reduces volume via segmentation into objects, but often underuses **relationships/context** between objects.
- Goal: provide a structured framework to (i) construct graphs from SITS and (ii) map graph representations to downstream tasks.

## What is the core method / protocol?

This paper is primarily **a review + design framework** rather than a single new model.

- Pipeline overview:
  1) Collect SITS datacubes.
  2) Create entities (nodes) at an object level (typically via segmentation / region delineation) and compute node features.
  3) Define inter-entity relationships (edges): spatial adjacency / proximity; temporal links; potentially spatio-temporal relations.
  4) Choose task family:
     - expert analysis (manual/visual graph analytics)
     - graph pattern mining
     - extrinsic prediction (e.g., classification/regression targets external to the graph)
     - intrinsic regression (imputation/forecasting within the graph)
  5) Use graph algorithms / graph learning (including GNNs) for inference.

- Emphasis on design trade-offs:
  - **Fidelity vs complexity**: segmentation granularity and edge density trade off representational faithfulness vs storage/time.

## What are the key metrics?

- As a survey/framework paper, metrics depend on case studies / cited works.
- The paper highlights practical considerations that often dominate in SITS:
  - predictive performance on downstream task (task-specific; e.g., accuracy/F1 for land-cover mapping; error metrics for forecasting)
  - computational cost (storage/time) as graph construction scales.

## What are the main results?

- Provides a consolidated conceptual map: from OBIA → graph-based SITS, including graph definitions, tasks, and design knobs.
- Includes two case studies (as described in the paper’s outline/abstract):
  - precise + dynamic land-cover mapping (local vs contextual information; pixel vs graph methods; segmentation effects)
  - water resources forecasting with GNNs (segmentation and time-series-length sensitivity)

## How is this similar to GALILEO?

- Both aim to make remote-sensing representations **usable across diverse tasks** rather than bespoke pipelines.
- Both emphasize exploiting **spatial + temporal structure** (here: explicit graphs; GALILEO: learned representations over imagery/time series).

## How is this different from GALILEO?

- This work is about **explicit graph construction and graph learning** (object nodes + designed edges), not a foundation model pretraining recipe.
- Focus is largely on *object-based structural modeling*; less on multi-modal pretraining, scaling, and general-purpose embeddings.

## Where GALILEO is stronger / cleaner (if true)

- Can avoid brittle segmentation/graph-design choices by learning directly from pixels/time series and using pretraining to amortize feature learning.
- Easier “drop-in” usage for many tasks if a unified embedding works well.

## Where GALILEO is weaker / needs to improve

- If downstream problems depend heavily on **explicit relational reasoning** (e.g., neighborhood effects, topological constraints), a pure embedding approach may lag unless it exposes/learns relational structure.
- Might need clearer story on when explicit object graphs (or hybrids) are beneficial.

## Action items for GALILEO (experiments / method / writing)

- [ ] Related-work framing: add a short paragraph contrasting *implicit structure learning* (foundation models) vs *explicit structure* (OBIA→graphs), and when each helps.
- [ ] Consider a hybrid baseline: Galileo embeddings as node features + lightweight graph layer for tasks with strong spatial interactions.
- [ ] In discussion/limitations, mention segmentation/graph-construction as an alternative path with different tradeoffs (interpretability vs pipeline fragility).

## Quotes / details to potentially cite

- Abstract-level motivation (graph methods “abandon the regular Euclidean structure … to work at an object level” and model “spatial and temporal interactions between identified objects”).
- OBIA motivation: object-level units + semantics; but context/relations can be critical for some phenomena.
