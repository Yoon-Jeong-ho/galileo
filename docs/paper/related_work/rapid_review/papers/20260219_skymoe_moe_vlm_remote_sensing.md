# SkyMoE: A Vision-Language Foundation Model for Enhancing Geospatial Interpretation with Mixture of Experts

- Year: 2025
- Venue: arXiv
- Authors: Jiaqi Liu, Ronghao Fu, Lang Sun, Haoran Liu, Xiao Yang, Weipeng Zhang, Xu Na, Zhuoran Duan, Bo Yang
- URL: https://arxiv.org/abs/2512.02517
- BibTeX key (if we add it): skymoe2025
- Tags: remote-sensing, vision-language, foundation-model, mixture-of-experts, multi-task, multi-granularity, benchmark

## One-sentence takeaway

SkyMoE is a remote-sensing VLM that uses a task/granularity-aware MoE router plus contrastive “local vs global” augmentation to improve multi-task, multi-granularity geospatial interpretation, evaluated on a new benchmark spanning many RS tasks.

## What problem does it solve?

- General-purpose VLMs (and many RS-VLMs) over-rely on global/background context and underperform on remote sensing tasks requiring both fine local instance details (e.g., counting small objects) and global scene context.
- Existing RS-VLMs tend to use unified/monolithic modeling that does not adapt to task type or interpretation granularity.
- Prior RS MoE approaches often do not enforce meaningful expert specialization (experts overlap).

## What is the core method / protocol?

- **MoE VLM architecture (“SkyMoE”)**:
  - Standard VLM stack: image encoder + visual adapter + decoder-only LLM.
  - Introduces **Mixture-of-Experts** in the LLM (FFN experts) and an **adaptive router** that generates **task- and granularity-aware routing instructions** to select experts.
- **Context-disentangled augmentation**:
  - Construct contrastive pairs intended to separate **local** and **global** cues.
  - Described as “systematically modifying local object attributes while preserving global context” to push experts toward level-specific representations.
- **Training (two-stage)** (as described in the HTML):
  - Stage I: pretrain multimodal understanding without MoE layers.
  - Stage II: specialize MoE (experts initialized by cloning FFN weights, then fine-tune for expert differentiation).
- **Benchmark**: introduces **MGRS-Bench**, covering multiple RS interpretation tasks and granularity levels; overall eval reports results across **21 public datasets** and “five” task categories.

## What are the key metrics?

- Paper reports “SOTA across tasks” over 21 datasets; likely a mix of task-specific metrics depending on task type (classification accuracy/F1, detection mAP, counting error, captioning/VQA style scores), but the arXiv HTML excerpt does not enumerate them.
- Qualitative diagnostic: masking objects and observing counting robustness (motivating analysis) suggests an emphasis on instance-level fidelity.

## What are the main results?

- Claims **state-of-the-art performance across tasks** on 21 datasets, with a radar-plot comparison against 8 SOTA models.
- Emphasizes improved adaptability across **task types** and **granularity levels** (local vs global interpretation).

## How is this similar to GALILEO?

- Both are in the orbit of **geospatial / remote-sensing foundation models** and evaluation across multiple downstream tasks.
- The “granularity” framing (local object details vs global context) is closely aligned with the common RS tension GALILEO likely must manage.
- Benchmarking breadth (many datasets) is similar in spirit to comprehensive geospatial model evaluation.

## How is this different from GALILEO?

- SkyMoE is a **vision-language** model with an **LLM decoder** and MoE routing; GALILEO may be more vision- or geospatial-encoder-centric (depending on the project), and may not rely on LLM expert routing.
- The novelty here is specifically **MoE specialization by task + granularity** and a **contrastive augmentation** for local/global disentanglement, rather than (e.g.) geospatial pretraining objectives, sensor fusion, or retrieval.
- SkyMoE introduces a **new benchmark (MGRS-Bench)**; GALILEO might target different benchmarks or problem formulations.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO is not LLM-heavy, it may be **simpler and cheaper** to train/infer than MoE LLM-based VLMs while still achieving strong performance on purely-visual RS tasks.
- If GALILEO has clearer geospatial inductive biases (projection geometry, multi-sensor alignment, temporal modeling), it may cover regimes SkyMoE does not emphasize.

## Where GALILEO is weaker / needs to improve

- If GALILEO struggles with **instruction-following** / **open-vocabulary** or multi-task language-supervised settings, SkyMoE’s VLM + router approach is a concrete template to close the gap.
- If GALILEO exhibits background-prior reliance (like the masking/counting example), it may need explicit mechanisms to separate local vs global cues.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add an explicit discussion (and possibly an ablation) of the **local-vs-global granularity tension** in RS interpretation; cite SkyMoE as motivation.
- [ ] Consider a **routing / conditional-computation** baseline (does not have to be full MoE LLM): e.g., lightweight expert heads specialized for local-object vs global-scene tasks.
- [ ] Add a **masking-based diagnostic**: measure whether predicted counts / detections change appropriately when foreground objects are occluded.
- [ ] If multi-task training, try **contrastive local-vs-global augmentations** (local perturbation with global context preserved) and test whether it improves instance-level metrics without hurting scene-level.

## Quotes / details to potentially cite

- “Existing geospatial VLMs typically adopt a unified modeling strategy and struggle to differentiate between task types and interpretation granularities, limiting their ability to balance local detail perception and global contextual understanding.”
- SkyMoE uses “an adaptive router that generates task- and granularity-aware routing instructions”.
- They introduce “a context-disentangled augmentation strategy that creates contrastive pairs between local and global features”.
- “Extensive experiments on 21 public datasets demonstrate that SkyMoE achieves state-of-the-art performance across tasks.”
