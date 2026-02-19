# VLM-3D: End-to-End Vision-Language Models for Open-World 3D Perception

- Year: 2025
- Venue: arXiv
- Authors: Fuhao Chang; Shuxin Li; Yabei Li; Lei He
- URL: https://arxiv.org/html/2508.09061
- BibTeX key (if we add it): vlm3d_chang_2025
- Tags: vision-language, 3d-detection, open-set, autonomous-driving, qwen2-vl, lora

## One-sentence takeaway

An early attempt to make a VLM (Qwen2-VL) directly output 3D boxes end-to-end for open-set driving perception, using LoRA fine-tuning and a staged “semantic then geometric (3D IoU)” loss schedule.

## What problem does it solve?

- Open-set / open-world perception in autonomous driving: detecting categories not seen during training, while still producing *3D* geometry (3D bounding boxes) suitable for downstream planning.
- Avoids multi-stage pipelines where a VLM provides features and a separate detector does geometry, which can propagate errors and prevents joint optimization.

## What is the core method / protocol?

- Backbone: Qwen2-VL (vision-language model).
- Training: parameter-efficient fine-tuning via LoRA inserted into (self-)attention modules.
- Output: model is trained to predict 3D bounding box parameters in LiDAR coordinates: center (x,y,z), size (l,w,h), yaw (theta).
- Loss schedule (their key idea):
  - Stage 1: token-level / “semantic” alignment loss (they describe MSE-style alignment between predicted vs GT representations) to stabilize training and convergence.
  - Stage 2: introduce 3D IoU loss to refine 3D box geometry; use weighted combination to shift focus to geometric accuracy.
- Evaluation: nuScenes; claims open-set generalization with qualitative examples on “unseen” categories.

## What are the key metrics?

- They report Accuracy / Recall / F1 over training epochs (unclear mapping to standard 3D detection metrics).
- They also report IoU/mIoU for categories (3D IoU).
- (Notably missing vs common driving 3D detection practice: mAP/NDS-style nuScenes metrics, detailed per-class AP, latency, ablations that isolate open-set behavior.)

## What are the main results?

- Claimed +12.8% improvement in “perception accuracy” from their joint semantic-geometric loss design (relative baseline not fully clear from the HTML text).
- Stage-2 training with IoU loss appears to stabilize and slightly improve their reported Accuracy/F1 vs stage-1-only.
- Shows qualitative visualizations of 3D boxes for open-set categories (e.g., construction worker, animal).

## How is this similar to GALILEO?

- Uses language / VLM priors to tackle open-world recognition rather than a fixed closed-set label space.
- Focuses on producing *geometric* outputs (3D boxes) suitable for embodied/AV pipelines, not just 2D grounding.
- Emphasizes end-to-end training as a way to avoid brittle modular perception stacks.

## How is this different from GALILEO?

- Very “direct regression” framing: prompt + image to 3D box parameters, with LoRA + loss scheduling as the main engineering.
- Appears to lean on camera images + nuScenes labels; unclear if it truly integrates multi-sensor fusion beyond coordinate transforms.
- Evaluation/reporting does not clearly follow standard open-vocabulary/open-set 3D detection benchmarks; open-set handling is more asserted than rigorously measured.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO has a clearer open-vocabulary protocol (text-conditioned class space, calibration, unknown handling criteria) and standard metrics/benchmarks, it will read as more rigorous than VLM-3D’s “accuracy/recall” tables.
- If GALILEO disentangles *recognition* vs *localization* vs *unknown detection*, it can offer a cleaner problem definition.

## Where GALILEO is weaker / needs to improve

- If GALILEO is still modular (VLM features + separate 3D head), VLM-3D is a useful “end-to-end” reference point to justify why tighter coupling might help.
- If GALILEO lacks a stable training recipe, the staged semantic->geometric loss schedule is a plausible stabilization trick to consider.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider citing VLM-3D as evidence of interest in end-to-end VLM-based 3D perception for open-world driving.
- [ ] If relevant, try a staged loss schedule: start with an easier semantic/alignment objective, then introduce geometry-aware losses (3D IoU / GIoU / corner losses) later.
- [ ] In related work, contrast GALILEO’s evaluation protocol against VLM-3D’s less-standard metrics; emphasize the need for standard nuScenes metrics + explicit open-set measurements.
- [ ] If GALILEO uses LoRA/PEFT, mention VLM-3D as precedent for PEFT on VLM backbones in AV perception.

## Quotes / details to potentially cite

- “We propose VLM-3D, the first end-to-end framework that enables VLMs to perform 3D geometric perception in autonomous driving scenarios.”
- “VLM-3D incorporates Low-Rank Adaptation (LoRA) … [and] introduces a joint semantic-geometric loss design … token-level semantic loss … while 3D IoU loss is introduced in later stages …”
- “Evaluations on the nuScenes dataset demonstrate … a 12.8% improvement in perception accuracy …”
