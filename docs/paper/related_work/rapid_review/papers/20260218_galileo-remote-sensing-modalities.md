# Galileo: Learning Global & Local Features of Many Remote Sensing Modalities

- Year: 2025
- Venue: arXiv (cs.CV)
- Authors: Gabriel Tseng; Anthony Fuller; Marlena Reil; Henry Herzog; Patrick Beukema; Favyen Bastani; James R. Green; Evan Shelhamer; Hannah Kerner; David Rolnick
- URL: https://arxiv.org/abs/2502.09356
- BibTeX key (if we add it): tseng2025galileo-remotesensing
- Tags: remote-sensing, multimodal, self-supervised, masked-modeling, multi-scale, contrastive

## One-sentence takeaway

A highly multimodal, self-supervised transformer learns shared multi-scale representations across diverse remote-sensing modalities via masked modeling plus dual global/local contrastive objectives, yielding a single generalist model that beats many specialist baselines across numerous benchmarks.

## What problem does it solve?

- Remote sensing needs shared representations over *heterogeneous* modalities (optical multi-spectral, SAR, elevation, weather, pseudo-labels, etc.) and *multi-scale* objects (tiny fast objects like boats vs. huge slow ones like glaciers).
- Existing approaches often specialize to a modality/task or struggle to capture both global context and local details across space/time.

## What is the core method / protocol?

- A multimodal transformer trained self-supervised over space-time remote sensing inputs.
- Masked modeling with flexible modality subsets.
- Two contrastive losses:
  - **Global contrastive loss** targets *deep representations* with **structured masking**.
  - **Local contrastive loss** targets *shallow input projections* with **unstructured masking**.
- Emphasis on extracting **global + local**, **multi-scale** features.

## What are the key metrics?

- Downstream benchmark performance across multiple remote-sensing tasks (paper claims 11 benchmarks), spanning:
  - Satellite image understanding
  - Pixel time-series tasks
- (Specific metrics are task-dependent; not captured from abstract alone.)

## What are the main results?

- Claims a **single generalist model** outperforms state-of-the-art specialist models across **11 benchmarks** and multiple tasks (satellite images + pixel time series).

## How is this similar to GALILEO?

- High-level: “generalist model” framing and emphasis on robustness/generalization across many settings.
- Otherwise mostly a **name collision**: this is remote-sensing representation learning, not LLM/agent evaluation.

## How is this different from GALILEO?

- Domain: earth observation / remote sensing (multimodal geospatial data) vs. GALILEO’s focus (LLM/agent-related work in this rapid-review queue).
- Method: self-supervised masked modeling + contrastive learning vs. GALILEO’s evaluation/agent methodology.
- Evaluation: remote-sensing downstream tasks/benchmarks vs. agent action/goal progress metrics.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO targets agent evaluation: clearer alignment to interactive multi-step decision making and tool use (this paper does not address that setting).

## Where GALILEO is weaker / needs to improve

- If GALILEO aims for “generalist” claims, this paper is a strong example of:
  - multi-modality handling
  - multi-scale representation learning
  - broad benchmark coverage

## Action items for GALILEO (experiments / method / writing)

- [ ] If the paper section needs disambiguation: add a brief note that “Galileo” is also used as a name for a remote-sensing foundation model (to avoid confusion when citing “Galileo” elsewhere).
- [ ] Consider whether GALILEO writing benefits from borrowing this paper’s framing: explicitly separating *global* vs. *local* signals and *structured* vs. *unstructured* perturbations/masking (as an analogy for evaluation axes).

## Quotes / details to potentially cite

- “a highly multimodal transformer to represent many remote sensing modalities … across space and time.”
- “objects of interest vary massively in scale, from small boats (1-2 pixels and fast) to glaciers (thousands of pixels and slow).”
- “dual global and local contrastive losses differ in their targets … and masking strategies ….”
- “outperforms SoTA specialist models … across eleven benchmarks ….”
