# Open3DTrack: Towards Open-Vocabulary 3D Multi-Object Tracking

- Year: 2025 (arXiv v2; originally 2024)
- Venue: arXiv
- Authors: Ayesha Ishaq, Mohamed El Amine Boudjoghra, Jean Lahoud, Fahad Shahbaz Khan, Salman Khan, Hisham Cholakkal, Rao Muhammad Anwer
- URL: https://arxiv.org/html/2410.01678
- BibTeX key (if we add it): open3dtrack_ishaq_2024
- Tags: open-vocabulary, 3d, multi-object-tracking, autonomous-driving, vision-language, tracking-by-detection

## One-sentence takeaway

A tracking-by-detection 3D MOT pipeline is made “open-vocabulary” by (i) training the 3D tracker to be class-agnostic on base classes and then (ii) labeling the resulting tracks using 2D image cues plus a pretrained vision-language model over both base+novel category prompts.

## What problem does it solve?

- Standard 3D multi-object tracking (MOT) benchmarks assume a closed set of object categories (car/pedestrian/cyclist, etc.).
- In real driving, a system encounters long-tail / novel objects; closed-vocab tracking fails because detections and track management depend on known class labels.
- Paper formalizes “open-vocabulary 3D tracking”: track objects in 3D over time while recognizing / labeling both seen (base) and unseen (novel) classes.

## What is the core method / protocol?

- Formulation: train on base category set C^base; test-time receives prompts / category list including C^novel (disjoint).
- Backbone tracker: adapts 3DMOTFormer (graph transformer for association) within a tracking-by-detection framework.
- Key design idea (system overview):
  - Use 3D proposals from base classes to train the tracker in a *class-agnostic* way (so association does not depend on closed-set semantics).
  - At inference, classify the 3D proposals / tracks using:
    - 2D image cues (multi-view projections of 3D boxes), and
    - a pretrained vision-language model (CLIP-like) over text prompts that include both base and novel class names.
- Includes dataset split design for different open-vocabulary scenarios (details not fully captured from skim; see paper tables).

## What are the key metrics?

- Uses standard autonomous-driving 3D MOT tracking-by-detection metrics (e.g., association quality and detection quality; typically AMOTA/AMOTP/IDS/IDF1/HOTA depending on benchmark).
- Also reports the *gap* between base-class tracking and novel-class tracking, and shows adaptation reduces this gap.

## What are the main results?

- Demonstrates that adding open-vocabulary labeling on top of a class-agnostic 3D tracker yields robust tracking on both known and novel categories in outdoor driving sequences.
- Claims to be the first paper explicitly addressing open-vocabulary 3D MOT with proposed evaluation splits.

## How is this similar to GALILEO?

- Uses pretrained foundation (vision-language) representations to generalize beyond the closed label space.
- Clear “separate geometry/association from semantics” pattern: track in 3D using geometric/temporal cues; attach semantics via a stronger open-vocab recognizer.
- Strongly aligned with a *prompt-time* generalization story (novel class list provided at test time).

## How is this different from GALILEO?

- Task/domain: autonomous driving 3D MOT (LiDAR + images), not GALILEO’s core setting.
- Approach is largely a *system integration* (class-agnostic tracker + open-vocab classifier) rather than an end-to-end jointly trained open-world model.
- Semantics comes from category-name prompts (discrete class set), not free-form language grounding or instruction following.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO is positioned as a more general, unified approach, it can emphasize fewer moving parts (vs. tracker + projection + VLM fusion + split engineering).
- Potentially richer language interface than fixed category prompts.

## Where GALILEO is weaker / needs to improve

- Open3DTrack gives a concrete benchmark story: explicit open-vocab *tracking* protocol + dataset splits; if GALILEO lacks an analogous evaluation protocol, this is a gap.
- If GALILEO needs to claim open-vocabulary temporal consistency, this paper is a direct prior that must be cited/positioned against.

## Action items for GALILEO (experiments / method / writing)

- [ ] Related-work: cite as “first open-vocabulary 3D MOT” and highlight the decomposition (class-agnostic tracking + VLM labeling).
- [ ] Writing: consider adopting their framing (C^base / C^novel, explicit split definitions) if GALILEO also evaluates generalization.
- [ ] Method: consider whether GALILEO should explicitly separate temporal association from semantic labeling (and discuss pros/cons).

## Quotes / details to potentially cite

- Abstract (task claim): “we introduce open-vocabulary 3D tracking… formulate the problem… introduce dataset splits… first to address open-vocabulary 3D tracking.”
- System overview (Fig. 2 description): tracker trained class-agnostic on base-class 3D proposals; inference classifies proposals/tracks using 2D cues + pretrained vision-language model over base+novel categories.
