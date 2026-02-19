# BoxFusion: Reconstruction-Free Open-Vocabulary 3D Object Detection via Real-Time Multi-View Box Fusion

- Year: 2025
- Venue: arXiv
- Authors: Yuqing Lan; Chenyang Zhu; Zhirui Gao; Jiazhao Zhang; Yihan Cao; Renjiao Yi; Yijie Wang; Kai Xu
- URL: https://arxiv.org/html/2506.15610
- BibTeX key (if we add it): boxfusion2025lan
- Tags: open-vocabulary, 3d-detection, rgb-d, online-perception, multi-view, fusion, reconstruction-free

## One-sentence takeaway

A reconstruction-free online open-vocabulary 3D detection pipeline that fuses per-frame RGB-D 3D box proposals (from a foundation model) into consistent global boxes via association + particle-filter-style random optimization, enabling real-time large-scale scanning.

## What problem does it solve?

- Existing (open-vocabulary) 3D detection and online perception pipelines often depend on dense point cloud / mesh reconstruction, which is memory- and compute-heavy and makes real-time deployment difficult, especially at large scale.
- Single-view 3D box predictors exist, but they are view-local and not robust as a full online scene-level detector without multi-view fusion.

## What is the core method / protocol?

- Input: streaming posed RGB-D frames (online scanning / embodied setting).
- Per-keyframe single-view proposals:
  - Use Cubify Anything (pretrained) to predict metric 3D bounding boxes from a single RGB-D frame.
  - Project predicted 3D boxes to the image to crop regions and extract open-vocabulary semantics using CLIP features.
- Multi-view association (to group boxes belonging to the same instance):
  - 3D NMS to remove redundant / overlapping boxes.
  - Additional correspondence matching (described as 2D box correspondence) intended to help associate small objects that may not overlap strongly in 3D.
- Multi-view box fusion (to obtain one global 3D box per object):
  - IoU-guided efficient random optimization framed as particle filtering.
  - Uses the IoU of convex hulls of projected 3D box corners as a multi-view consistency objective (optimize a fused box that better matches the set of view-specific proposals).
  - Mentions pre-sampled particle swarm templates for online efficiency.

## What are the key metrics?

- Online open-vocabulary 3D detection accuracy on:
  - CA-1M
  - ScanNetV2
- System metrics emphasized:
  - Real-time throughput (claims >20 FPS)
  - GPU memory usage (claims ~7 GB)
  - Robustness / scalability to very large environments (>1000 m^2)

## What are the main results?

- Claims state-of-the-art among online methods on CA-1M and ScanNetV2.
- Emphasizes a practical operating point: real-time (>20 FPS) and relatively low GPU memory (~7 GB), while handling large-scale spaces (multi-floor, >1000 m^2) without dense reconstruction.

## How is this similar to GALILEO?

- Strongly aligned with an object-centric, sparse scene representation mindset: bounding boxes + semantics as sufficient structure for downstream embodied tasks.
- Uses foundation model outputs as primitives (per-frame proposals + language-aligned semantics) and then performs geometric / temporal fusion to build a coherent world model.

## How is this different from GALILEO?

- Task framing is specifically online open-vocabulary 3D object detection from posed RGB-D streams (not a general world model / agent pipeline).
- Depends on a particular single-view 3D box proposal model (Cubify Anything) and then focuses on multi-view consistency via association + optimization.
- The fusion objective is explicitly geometric (IoU via projected box corners) and solved by random/particle-filter optimization rather than (e.g.) learned end-to-end multi-view fusion.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO already maintains a unified scene graph / map beyond boxes (relations, persistence, uncertainty, task-conditioned querying), BoxFusion is a narrower perception module.
- If GALILEO avoids dependence on a specialized RGB-D box foundation model, it may generalize better across sensors/modalities.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently relies on dense reconstruction or heavy volumetric mapping for perception, this paper is a clear alternative paradigm worth benchmarking against.
- If GALILEO lacks an explicit, lightweight multi-view box-level fusion mechanism with real-time constraints, BoxFusion suggests a concrete design.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding (or at least discussing) a reconstruction-free perception baseline: per-frame object proposals + multi-view box fusion.
- [ ] If we have multi-view observations, test a simple association+fusion pipeline vs. heavier mapping to quantify compute/memory tradeoffs.
- [ ] In related work, explicitly cite the idea that sparse object-layout representations can be sufficient for embodied tasks (the paper motivates this claim).
- [ ] If relevant, replicate the paper’s fusion idea: IoU-guided random optimization / particle-filter-style update as a fast online refinement step.

## Quotes / details to potentially cite

- “Existing detection methods, whether offline or online, typically rely on dense point cloud reconstruction, which imposes substantial computational overhead and memory constraints, hindering real-time deployment…”
- Claims: real-time performance “over 20 FPS” and “7GB GPU memory usage” even in environments “exceeding 1000 square meters.”
- Pipeline components: Cubify Anything (single-view 3D box proposals) + CLIP (open-vocabulary semantics) + association (3D NMS + correspondence) + IoU-guided random optimization based on particle filtering for multi-view box fusion.
