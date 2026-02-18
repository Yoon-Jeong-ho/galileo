# Geospatial Foundation Models to Enable Progress on Sustainable Development Goals

- Year: 2025
- Venue: arXiv
- Authors: Weikang Yu; Xiaokang Zhang; Aldino Rizaldy; Jian Wang; Chufeng Zhou; Richard Gloaguen; Gustau Camps-Valls
- URL: https://arxiv.org/abs/2505.24528
- BibTeX key (if we add it): yu2025sustainfm
- Tags: geospatial, foundation-model, earth-observation, benchmarking, sdg, domain-shift, energy

## One-sentence takeaway

SustainFM is an SDG-grounded benchmark suite (16 datasets across diverse EO tasks) arguing geospatial FM evaluation should include transfer/generalization and energy efficiency, not just accuracy.

## What problem does it solve?

- The geospatial FM ecosystem is growing fast, but comparisons are fragmented and often not aligned with real-world “impact” goals (here: UN SDGs).
- Common FM evaluations under-emphasize (a) cross-dataset / cross-task transfer, (b) robustness to domain shift, and (c) compute/energy cost—key for sustainability-aligned deployment.

## What is the core method / protocol?

- Proposes **SustainFM**, a benchmarking framework grounded in the **17 SDGs**.
- Curates a collection of **16 datasets** aligned with SDG1–SDG16 (SDG17 = partnerships framing), spanning:
  - multiple continents,
  - spatial resolutions (reported range: ~0.5m to 30m),
  - diverse tasks (example given: **asset wealth prediction** to **environmental hazard detection**).
- Uses SustainFM as an empirical lens to compare geospatial foundation models vs “traditional approaches” across tasks/datasets.
- Emphasizes evaluation axes beyond raw accuracy, including transferability/generalization and energy efficiency.

## What are the key metrics?

- Task metrics (varies by dataset; paper positions this as “accuracy” or task performance).
- **Transferability / generalization** across datasets/tasks.
- **Energy efficiency / compute footprint** (advocated as a primary criterion for responsible use).
- **Robustness to domain shifts** (explicitly highlighted as important for deployment).

## What are the main results?

- FMs are **often** (not always) better than traditional baselines across diverse geospatial tasks/datasets.
- The paper argues for a broadened evaluation protocol: accuracy alone is insufficient; include transfer/generalization and energy.
- Positions geospatial FMs as enablers of scalable SDG-grounded solutions, with the caveat that evaluation should be impact-driven and ethically informed.

## How is this similar to GALILEO?

- Same overarching setting: **geospatial / EO foundation models** intended to transfer across tasks.
- Shares the framing that *generalization across heterogeneous data/tasks* is the core promise.

## How is this different from GALILEO?

- This work is **benchmarking + evaluation framing** (SustainFM), not a new representation model.
- Primary novelty is the **SDG-grounded suite** and the argument for **impact-driven deployment** criteria (incl. energy/robustness/ethics).

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO provides a unified multi-modal pretraining recipe/architecture, it can be positioned as a concrete instantiation that SustainFM can evaluate (i.e., GALILEO contributes the “model”; SustainFM contributes the “yardstick”).

## Where GALILEO is weaker / needs to improve

- If GALILEO’s evaluation is currently heavy on in-distribution metrics, this paper suggests adding:
  - explicit domain-shift tests,
  - transfer/few-shot protocols,
  - energy/compute reporting.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a related-work paragraph framing SustainFM as an SDG/impact-driven benchmark for geospatial FMs.
- [ ] If feasible, run (or at least discuss) evaluation of GALILEO-style representations on SustainFM-like axes:
  - cross-task transfer,
  - domain shift,
  - energy/compute reporting (training and inference).
- [ ] In the paper’s evaluation section, explicitly justify metric choices beyond accuracy (transfer + efficiency).

## Quotes / details to potentially cite

- “We introduce **SustainFM**, a comprehensive benchmarking framework grounded in the **17 Sustainable Development Goals** with extremely diverse tasks ranging from **asset wealth prediction** to **environmental hazard detection**.” (abstract)
- “Evaluating FMs should go beyond accuracy to include **transferability, generalization, and energy efficiency** as key criteria…” (abstract)
- SustainFM description: “collection of **16 datasets** … six continents … spatial resolutions ranging from **0.5 m to 30 m** …” (Figure 1 caption in arXiv HTML)
