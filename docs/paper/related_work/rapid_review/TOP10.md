# Rapid Review — Top 10 shortlist (rolling)

This is a rolling shortlist of the **10 most important adjacent papers**.

For each item, we maintain:
- What it contributes
- What it misses relative to GALILEO
- What we should borrow
- How we should change GALILEO (experiments/methods/claims)

> NOTE: This list is expected to change frequently until ~30–50 papers are read.

## Shortlist

1) **AnySat: One Earth Observation Model for Many Resolutions, Scales, and Modalities** (Astruc et al., arXiv 2024/2025)
   - Contributes: a unified **multi-sensor, multi-resolution** EO foundation model trained self-supervised with a **JEPA-style** objective + **scale-adaptive spatial encoders**, and a clear “one model across heterogeneous sensors” evaluation story (GeoPlex + external datasets).
   - Misses vs GALILEO: unclear (from abstract alone) how well it handles *long temporal sequences* / richer modality alignment beyond the GeoPlex set; details of objectives/ablations and what truly drives transfer are not visible on the abstract page.
   - Borrow: (i) explicit **scale-adaptation** framing/mechanism, (ii) “heterogeneous sensor suite” narrative with a named benchmark collection, (iii) broad downstream coverage across mapping/classification/segmentation.
   - How to change GALILEO: add an explicit cross-resolution/scale-transfer evaluation slice and tighten the “one model across many sensors/datasets” claim with a compact cross-dataset table.

2) **Any-Optical-Model: A Universal Foundation Model for Optical Remote Sensing** (Hong et al., AAAI 2026; arXiv 2025)
   - Contributes: an optical RSFM explicitly designed to handle **arbitrary band compositions** and **wide resolution ranges** via (i) **per-band tokenization + band identity embeddings** and (ii) **resolution-adaptive patch embedding** (multi-kernel bank + pseudo-inverse resize).
   - Misses vs GALILEO: focused on optical multispectral (not a general multi-modal geospatial model); unclear from rapid read whether it supports temporal modeling beyond standard pretraining augmentations.
   - Borrow: (i) missing-band and cross-sensor evaluation slices as first-class claims, (ii) explicit band-identity encoding baseline, (iii) multi-scale semantic alignment framing.
   - How to change GALILEO: add explicit **missing-band / cross-sensor / cross-resolution** stress tests and (if needed) a tokenizer/pathway that preserves band identity while still allowing cross-band interaction.

3) **TiMo: Spatiotemporal Foundation Model for Satellite Image Time Series** (Qin et al., arXiv 2025)
   - Contributes: a **hierarchical** SITS foundation model with a custom attention pattern (**spatiotemporal gyroscope attention**) aimed at capturing multiscale aligned space-time structure; plus a temporally richer pretraining set (**MillionST**, 10 timestamps over 5 years at 100k locations).
   - Misses vs GALILEO: appears specialized to *aligned* SITS (Sentinel-2) and masked modeling; unclear how it extends to heterogeneous sensors/modalities or misaligned/missing time series.
   - Borrow: (i) “temporal cardinality matters” narrative and dataset design, (ii) structured attention that respects SITS alignment, (iii) hierarchical/pyramid backbone as a strong default for multiscale geospatial phenomena.
   - How to change GALILEO: add at least one baseline/ablation contrasting **flat ViT vs hierarchical** backbones for time series, and explicitly report transfer vs number of timestamps.

4) **Measuring Sycophancy of Language Models in Multi-turn Dialogues** (Hong et al., Findings of EMNLP 2025)
   - Contributes: concrete multi-turn social-pressure protocol + two extremely usable metrics: **Turn of Flip (ToF)** (time-to-failure proxy) and **Number of Flips (NoF)** (instability/oscillation).
   - Misses vs GALILEO: does not explicitly separate *evidence-driven belief revision* from *pressure-driven drift*; limited focus on *recovery after flip*.
   - Borrow: ToF/NoF framing and third-person prompting baseline (reported up to 63.8% reduction in debate).
   - How to change GALILEO: map our metrics onto a ToF/NoF-style vocabulary (or add direct analogues) and highlight our additional controls + recovery measurements.

5) **ELEPHANT: Measuring and understanding social sycophancy in LLMs** (Cheng et al., arXiv 2025)
   - Contributes: a theory-grounded expansion from “explicit agreement” → **social sycophancy as face-preservation** (validation, indirectness, framing, moral inconsistency), plus an evaluation showing LLMs preserve user face far more than humans in advice/wrongdoing contexts.
   - Misses vs GALILEO: does not center multi-turn *time-to-failure* / recovery dynamics; relies on human baselines and social-dimension classifiers rather than drift-vs-revision controls.
   - Borrow: the face-preservation taxonomy; the *stance-swap* moral-consistency probe (user adopts either side) as a clean way to reveal side-dependent affirmation.
   - How to change GALILEO: add “face channels” of pressure as first-class metrics/labels (validation/framing adoption), not only explicit belief agreement; include a moral-consistency or stance-swap subtest.

6) **Sycophancy Hides Linearly in the Attention Heads** (Genadi et al., arXiv 2026)
   - Contributes: a mechanistic localization claim that **correct→incorrect sycophancy** is most *steerable* via a **sparse subset of middle-layer attention heads**, plus evidence those heads attend disproportionately to **user doubt** cues.
   - Misses vs GALILEO: intervention relies on **internal activation access** + per-model head selection; less emphasis on long-horizon survival/recovery metrics.
   - Borrow: (i) the “**separable vs steerable**” distinction, (ii) head-level sparsity as a hypothesis about where social-pressure cues route.
   - How to change GALILEO: add an interpretability appendix: test whether pressure-driven flips are linearly decodable and whether a small set of attention heads dominates.

7) **T3: Benchmarking Sycophancy and Skepticism in Causal Judgment** (Chang, arXiv 2026)
   - Contributes: a clean **sensitivity vs specificity** decomposition (Utility vs Safety; Sheep vs Wolves), plus explicit measurement of **calibrated abstention** and **multi-turn flip dynamics** (Good Flip vs Bad Flip) under social + epistemic pressure.
   - Misses vs GALILEO: centered on causal-judgment vignettes rather than generic belief drift/revision; limited focus on recovery trajectories.
   - Borrow: Utility/Safety reporting; GoodFlip/BadFlip asymmetry.
   - How to change GALILEO: add an explicit endorse-true vs reject-false breakdown and track good/bad flip asymmetry.

8) **CausalT5K: Diagnosing and Informing Refusal for Trustworthy Causal Reasoning of Skepticism, Sycophancy, Detection-Correction, and Rung Collapse** (Geng et al., arXiv 2026)
   - Contributes: a large diagnostic benchmark with paired neutral vs pressure variants enabling flip-quality metrics + Utility/Safety decomposition.
   - Misses vs GALILEO: primarily causal-reasoning infrastructure; less about full recovery-after-flip trajectories.
   - Borrow: pressure-paired benchmark design + 2D reporting.
   - How to change GALILEO: report neutral-vs-pressure paired outcomes and add flip-quality rates.

9) **Time-To-Inconsistency: A Survival Analysis of Large Language Model Robustness to Adversarial Attacks** (Li, Krishnan, Padman, arXiv 2025)
   - Contributes: survival-analysis framing for multi-turn robustness with censoring; hazards/survival curves; **C-index** and **Integrated Brier Score**.
   - Misses vs GALILEO: evaluates adversarial inconsistency rather than persuasion; no explicit recovery-after-flip objective.
   - Borrow: time-to-event reporting + calibration-aware trajectory evaluation.
   - How to change GALILEO: add a survival-analysis reporting layer (hazard + censored survival).

10) **Modeling and Predicting Multi-Turn Answer Instability in Large Language Models** (He et al., arXiv 2025)
   - Contributes: re-questioning protocol + **stationary (long-run) accuracy** via a Markov-chain view of correctness transitions.
   - Misses vs GALILEO: limited controls for evidence-vs-pressure; stationary metrics can hide oscillation structure.
   - Borrow: stationary-accuracy framing as an interactive robustness headline metric.
   - How to change GALILEO: add stationary truth-rate as an additional reporting lens.

## Changelog

- 2026-02-19: Added *TiMo: Spatiotemporal Foundation Model for Satellite Image Time Series* (arXiv 2025) as a key **SITS** neighbor (hierarchical backbone + structured spatiotemporal attention; MillionST with 10 timestamps over 5 years). Displaced *Persuasion Dynamics in LLMs: Investigating Robustness and Adaptability in Knowledge and Safety with DuET-PD* (Tan et al., EMNLP 2025): useful dual-axis persuasion framing, but less central if the related-work emphasis is shifting toward EO foundation models.
- 2026-02-19: Added *Any-Optical-Model: A Universal Foundation Model for Optical Remote Sensing* (AAAI 2026; arXiv 2025) as a key **EO foundation model** neighbor focused on missing-band + cross-sensor + cross-resolution robustness. Displaced *Consistency of Large Reasoning Models Under Multi-Turn Attack* (Li, Krishnan, Padman, arXiv 2026): useful trajectory taxonomy, but less central if the related-work emphasis is shifting toward EO foundation models.
- 2026-02-19: Added *AnySat: One Earth Observation Model for Many Resolutions, Scales, and Modalities* (arXiv 2024/2025) as a key **geospatial foundation model** neighbor (multi-resolution + JEPA + heterogeneous sensors). Displaced *BASIL: Bayesian Assessment of Sycophancy in LLMs*: useful for drift-vs-revision controls, but less central if GALILEO’s related-work emphasis is shifting toward EO foundation models.
