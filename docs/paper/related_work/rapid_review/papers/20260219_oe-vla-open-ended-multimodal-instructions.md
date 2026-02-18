# Unveiling the Potential of Vision-Language-Action Models with Open-Ended Multimodal Instructions

- Year: 2025
- Venue: arXiv
- Authors: Wei Zhao; Gongsheng Li; Zhefei Gong; Pengxiang Ding; Han Zhao; Donglin Wang
- URL: https://arxiv.org/html/2505.11214v1
- BibTeX key (if we add it): oe-vla-zhao-2025
- Tags: vla, multimodal-instructions, calvin, llava-next-interleave, action-tokenization

## One-sentence takeaway

OE-VLA extends standard vision-language-action policies to accept open-ended *multimodal* instructions (goal image, object image, optical text-in-image, short video demo) via a unified interleaved VLM backbone, and benchmarks this on two CALVIN-derived suites.

## What problem does it solve?

- Most VLA systems assume the human prompt is only natural language, but real HRI often provides goals via images (show-me-this), text in the scene (whiteboard), a goal-state image, or a short demonstration video.
- Need a single policy that can interpret these different instruction types without separate per-task models.

## What is the core method / protocol?

- Model: a VLM backbone that can handle interleaved multi-image + text input (they choose LLaVA-Next-Interleave) with:
  - Vision encoder: SigLIP ViT
  - Projector: 2-layer MLP into LLM hidden space
  - LLM: Qwen-1.5 backbone (32k context)
- Action representation: discretize continuous robot actions into 256 bins and reuse rare LLM tokens as action tokens; predict a 5-step action chunk autoregressively (no diffusion/policy head).
- Training data construction: convert existing language-annotated robot data into 5 subsets:
  - Keep original language instructions (to preserve standard VLA ability)
  - Replace object mentions with cropped object images (visual object specification)
  - Render text instructions into images with varying styles (optical instruction following)
  - Replace instruction with a few frames from the trajectory (video demo learning)
  - Provide a goal image (visual goal reaching)
- Two-stage curriculum:
  1) Multi-image grounding finetune (MGrounding) to improve spatial/multi-image perception
  2) Open-ended instruction tuning on the constructed multimodal robot data

## What are the key metrics?

- CALVIN long-horizon success breakdown (LH-1..LH-5) and average successful sequence length (Len).
- Performance on two new benchmarks:
  - OE-CALVINbase: open-ended instructions from same environment / easier distributions
  - OE-CALVINhard: web images, handwritten styles, diverse viewpoints / harder distributions

## What are the main results?

- On CALVIN (language-only prompts), OE-VLA remains competitive vs classic language-conditioned baselines.
- On open-ended instruction benchmarks:
  - Strongest on visual object specification (object-image prompts).
  - Optical instruction following and video-demo learning are "satisfactory".
  - Visual goal reaching (single goal image) is hardest and degrades most.
- Scaling from ~1B to ~7B improves robustness on the harder open-ended benchmark and reduces the gap to language-only performance.

## How is this similar to GALILEO?

- Same broad thrust: generalist robot policies that follow *rich* task specifications (beyond plain text) and evaluate generalization.
- Uses an interleaved multimodal foundation model as the policy backbone.

## How is this different from GALILEO?

- Focus is specifically "open-ended instruction modalities" for VLA on CALVIN; action modeling is via discretized tokens and does not incorporate a separate continuous action head (e.g., diffusion/flow).
- Their open-ended modalities are defined as four categories (object image, text-in-image, goal image, short demo frames) and mostly derived by transforming existing datasets.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO uses a more principled continuous action modeling (diffusion/flow or structured action spaces), it may avoid discretization artifacts and potentially improve fine control.
- If GALILEO includes stronger grounding/temporal world modeling, it may better address the "visual goal reaching" failure mode.

## Where GALILEO is weaker / needs to improve

- Need to ensure GALILEO’s instruction interface covers these concrete modalities (goal image, object image, scene text, short demo) in a unified input format and evaluation protocol.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add/describe an "open-ended instruction" suite mirroring: object-image specification, text-in-image instructions, goal-image reaching, short demo learning.
- [ ] Include an ablation: curriculum pretraining on multi-image grounding vs not.
- [ ] Call out that goal-image reaching is the hardest; discuss what architectural component would address it (temporal reasoning, state-diff, planning).

## Quotes / details to potentially cite

- Motivation: prior VLAs "usually accept only one form of human prompting, language instructions" and this constrains open-ended HRI.
- OE task taxonomy introduced: visual object specification (VOS), optical instruction following (OIF), visual goal reaching (VGR), video demo learning (VDL).
- Two-stage curriculum: (1) multi-image grounding finetune (MGrounding) then (2) open-ended instruction tuning.
