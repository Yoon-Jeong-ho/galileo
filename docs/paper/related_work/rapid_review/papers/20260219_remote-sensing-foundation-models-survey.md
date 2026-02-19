# A Survey on Remote Sensing Foundation Models: From Vision to Multimodality

- Year: 2025
- Venue: arXiv
- Authors: Ziyue Huang, Hongxi Yan, Qiqi Zhan, Shuai Yang, Mingming Zhang, Chenkai Zhang, YiMing Lei, Zeming Liu, Qingjie Liu, Yunhong Wang
- URL: https://arxiv.org/abs/2503.22081
- BibTeX key (if we add it): huang2025survey_remote_sensing_foundation_models
- Tags: remote-sensing; foundation-model; multimodal; survey

## One-sentence takeaway

A broad survey of vision and multimodal **remote sensing foundation models**, emphasizing modality fusion (optical/SAR/LiDAR + text/geo), datasets/tasks, and open challenges (alignment, transfer, scalability).

## What problem does it solve?

- Provides an organizing view of a fast-moving space: RS foundation models across modalities and tasks.
- Summarizes (claimed) recurring bottlenecks for practical deployment: heterogenous sensors, dataset/label scarcity, multimodal fusion complexity, and compute.

## What is the core method / protocol?

- Survey / taxonomy paper (not a new model).
- Focus areas called out in the abstract:
  - architectures for vision + multimodal RS foundation models
  - training methods (self-supervised / semi-supervised / multimodal learning)
  - datasets + application scenarios
  - challenges: data alignment, cross-modal transfer, scalability
- Maintains a public resource list: https://github.com/IRIP-BUAA/A-Review-for-remote-sensing-vision-language-models

## What are the key metrics?

- Not specified in the abstract (survey). Likely reports task metrics commonly used in RS:
  - detection (mAP), segmentation (mIoU/F1), classification (acc/F1), change detection (F1/IoU), etc.

## What are the main results?

- Not an empirical “result” paper; main contribution is synthesis:
  - argues multimodal integration improves RS tasks (detection, land-cover classification, change detection)
  - enumerates limitations: annotation scale, fusion complexity, compute demands
  - highlights research directions around alignment/transfer/scaling

## How is this similar to GALILEO?

- Shares the “foundation model for geospatial/remote-sensing understanding” framing.
- Mentions multimodal settings relevant to geospatial intelligence (sensor fusion + text/geo context).

## How is this different from GALILEO?

- This is a survey/taxonomy; it does not propose a concrete training objective, benchmark protocol, or intervention.
- Emphasis appears to be on RS perception tasks and multimodal fusion engineering, rather than any GALILEO-specific evaluation paradigm.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO offers a crisp protocol/benchmark or a focused mechanism claim, it should be easier to evaluate than a broad survey narrative.

## Where GALILEO is weaker / needs to improve

- If GALILEO is not explicitly positioned against the RS multimodal literature (optical/SAR/LiDAR + text), this survey is a reminder to situate the work in that ecosystem.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider citing this as an “RS foundation models landscape” reference if we need a broad umbrella citation.
- [ ] Cross-check whether it names a standard set of datasets/tasks we should mention (or explicitly justify excluding).
- [ ] If we claim novelty in multimodal geospatial fusion, ensure we contrast with the main categories highlighted here (alignment, cross-modal transfer, scalability).

## Quotes / details to potentially cite

- “These models combine various data modalities, such as optical, radar, and LiDAR imagery, with textual and geographic information…”
- Key challenges listed: “data alignment, cross-modal transfer learning, and scalability.”
