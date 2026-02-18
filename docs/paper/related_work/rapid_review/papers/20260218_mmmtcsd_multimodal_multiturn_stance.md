# Multimodal Multi-turn Conversation Stance Detection: A Challenge Dataset and Effective Model

- Year: 2024
- Venue: ACM Multimedia (MM ’24)
- Authors: Fuqiang Niu et al.
- URL: https://arxiv.org/abs/2409.00597
- BibTeX key (if we add it): niu2024mmmtcsd
- Tags: stance-detection, multimodal, multi-turn, reddit, dataset, mllm

## One-sentence takeaway

Introduces MmMtCSD, a 21k-instance Reddit-based multimodal multi-turn conversational stance dataset, and a LoRA-tuned LLaMA2+ViT “MLLM-SD” fusion model that outperforms strong text-only and multimodal baselines.

## What problem does it solve?

- Existing multimodal stance detection (MSD) benchmarks largely treat each example as an isolated text-image pair, ignoring multi-party conversational context.
- Conversational stance often depends on prior turns (and sometimes the image), making context-free MSD under-specified.

## What is the core method / protocol?

- **Dataset (MmMtCSD):** Reddit posts + images + multi-turn comment threads; stance labels are {favor, against, none} for targets including **Tesla**, **Bitcoin**, and a **Post-as-target (Post-T)** setting.
- **Annotation:** multiple expert annotators; at least two per instance with adjudication; reports Cohen’s kappa (favor/against focus) per target.
- **Model (MLLM-SD):**
  - Text encoder builds a conversation-style prompt (includes one-shot chain-of-thought style example).
  - Adds **image captions** (generated with GPT-4V) to better align vision+text.
  - Visual encoder: ViT features projected to match LLM dims.
  - Fusion: LoRA fine-tuning of **LLaMA2-chat 7B** to integrate modalities; output mapped to stance labels.

## What are the key metrics?

- F1-avg = average of F1 for **against** and **favor** (ignores “none” in the averaging), following common stance-detection practice.
- Reports in-target and cross-target setups; also analyzes performance vs. conversation depth.

## What are the main results?

- MmMtCSD size: **21,340** labeled instances; **~66%** marked vision-related.
- In-target: MLLM-SD reports higher F1-avg than BERT/RoBERTa, Branch-BERT/GLAN, and multimodal baselines (BERT+ViT, TMPT, GPT4-Vision).
- Ablations: removing **captions** or **CoT prompt structure** degrades performance noticeably.

## How is this similar to GALILEO?

- Both care about **multi-turn conversational signals** (stance/behavior can be contextual rather than single-utterance).
- Emphasizes that evaluation should stress **context dependence** and **hard cases** rather than only standalone inputs.

## How is this different from GALILEO?

- Task is supervised **stance classification** with curated targets (Tesla/Bitcoin/Post-T), not persona/assistant stability per se.
- Uses a relatively direct multimodal fusion approach (caption + ViT + LoRA LLaMA2) rather than explicit stability objectives or longitudinal persona constraints.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO targets assistant/persona stability, it can frame “stance” more broadly (preferences/values/identity consistency) beyond stance-toward-target classification.
- Could avoid reliance on external captioning (GPT-4V) by standardizing vision-language encoders or reporting costs/latency.

## Where GALILEO is weaker / needs to improve

- If GALILEO lacks a **multimodal, multi-turn** benchmark slice, MmMtCSD is a concrete example showing the value of that setting.
- MmMtCSD explicitly measures depth effects; if GALILEO doesn’t stratify by context length/depth, it may miss systematic failure modes.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a related-work paragraph: “multimodal multi-turn conversational stance detection” as evidence that **context + vision** materially changes stance inference.
- [ ] Consider an analysis table/figure stratifying GALILEO outcomes by **conversation depth** (analogous to their depth 1 vs 2–4 vs 5–6).
- [ ] When discussing multimodal context, cite their dataset stats (21,340; 66% vision-related) as motivation that images frequently matter in real threads.

## Quotes / details to potentially cite

- Dataset claim: “MmMtCSD contain a total of **21,340** annotated data.”
- Vision relevance: total **14,083 / 21,340 (65.99%)** marked vision-related.
- Targets: Tesla, Bitcoin, and **Post-T** (posts as targets; diverse targets).
- Agreement: kappa reported as **0.72 (Bitcoin)**, **0.81 (Tesla)**, **0.68 (Post-T)**.
