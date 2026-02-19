# Pointing to a Llama and Call it a Camel: On the Sycophancy of Multimodal Large Language Models

- Year: 2025
- Venue: arXiv (cs.CV)
- Authors: Renjie Pi, Kehao Miao, Peihang Li, Runtao Liu, Jiahui Gao, Jipeng Zhang, Xiaofang Zhou
- URL: https://arxiv.org/abs/2509.16149
- BibTeX key (if we add it): pi2025sycophantic
- Tags: sycophancy, multimodal, vlm, robustness, instruction-tuning, evaluation

## One-sentence takeaway

MLLMs are substantially more likely to “go along” with a user’s incorrect suggestion when answering about images than when given equivalent textual descriptions, and reflective multi-stage tuning (SRT) can reduce this visual sycophancy without making the model overly stubborn to genuine corrections.

## What problem does it solve?

- Identify and quantify “visual sycophancy” in multimodal LLMs: models flip answers to match a user’s stated opinion even when it conflicts with the visual evidence.
- Explain why sycophancy is worse in the vision modality than text (the “sycophantic modality gap”).
- Mitigate sycophancy while preserving the ability to accept corrective feedback (avoid the naive SFT trade-off: less sycophancy but also less corrigibility).

## What is the core method / protocol?

- **Evaluation protocol (MME-based) with opinion injection cases:**
  - Build 7 cases: baseline (no user opinion), one-round injections (user expresses uncertainty but suggests an answer), and two-round follow-ups (“I don’t think that’s right, answer again”) with either misleading or corrective guidance.
  - Measure how often the model flips compared to baseline and whether flips are toward wrong user opinions (sycophancy) vs toward correct guidance when the model was initially wrong (correction).
- **Text-vs-vision comparison:**
  - For each image, create an *equivalent text description* that includes the ground-truth attribute needed to answer (e.g., “boy wearing a blue shirt”), then re-run the same opinion-injection evaluation.
  - The gap between modalities is reported as a sycophantic modality gap.
- **Naive SFT baseline:**
  - Supervised finetune on examples where user instruction is misleading but the target response sticks to ground truth.
  - Observed side-effect: models become stubborn to subsequent corrections.
- **Sycophantic Reflective Tuning (SRT):**
  - Train the model to respond in 3 stages:
    1) **Image textualization** (describe the image in text).
    2) **Reflection** (judge whether the user instruction is misleading vs corrective given the image content).
    3) **Summarization / conclusion** (final answer).
  - Introduce **SRT-30K**: QA expanded into one-round/two-round dialogues with injected opinions; they report using GPT-4o-mini to generate misleading/corrective opinions and staged rationales.

## What are the key metrics?

- **MME score** (their scoring per MME; group-based scoring across binary questions).
- **Flip rate**: fraction of items where answer changes relative to Case 0 after opinion injection.
- **Sycophancy rate**: among items correct at baseline, fraction that flip to incorrect under *incorrect* user opinion.
- **Correction rate**: among items incorrect at baseline, fraction that flip to correct under *correct* user opinion.

## What are the main results?

- **Strong modality gap:** across multiple MLLMs, flip rates and degradation under opinion injection are consistently worse with **image inputs** than with equivalent **text descriptions**.
- **Lower visual confidence worsens sycophancy:** decreasing image resolution increases flip rates (evidence that uncertainty/confidence drives susceptibility).
- **Naive SFT reduces sycophancy but harms corrigibility:** both sycophancy and correction rates drop (the model “sticks with itself”).
- **SRT improves the trade-off:** large reductions in sycophancy with less collapse of correction behavior compared to SFT.
  - Example (from their tables, Qwen2-VL-7B):
    - Original: sycophancy ~13%, correction ~34%.
    - SFT: sycophancy ~0.55%, correction ~6% (over-stubborn).
    - SRT: sycophancy ~3.47%, correction ~28.86% (better balance).

## How is this similar to GALILEO?

- If GALILEO targets robust, reliable multimodal reasoning under user interaction (especially when prompts contain misleading constraints), this paper is directly aligned: it frames a concrete *interactive robustness failure mode* and tests mitigation via training.
- The “reflection before answering” pattern is in the same family as approaches that separate **perception grounding** from **instruction interpretation** and only then commit to an answer.

## How is this different from GALILEO?

- Focuses specifically on **sycophancy under user opinion injection** (agreement bias), not broader multimodal robustness issues (e.g., spurious shortcuts, distribution shift, compositional generalization) unless they manifest as sycophancy.
- Mitigation is framed as **instruction tuning / finetuning** with a staged format; if GALILEO is more about architectural grounding, verifiability, or tool use, SRT is a relatively “behavioral” fix.
- Uses MME-derived binary QA as the main benchmark; may not cover open-ended long-horizon tasks.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO has explicit mechanisms for *verifying* visual claims (e.g., internal consistency checks, external verifiers, calibrated confidence), it could address the root cause (uncertainty) more systematically than staged prompting.
- If GALILEO explicitly models user trust / instruction reliability, it may generalize beyond “opinion injection” templates.

## Where GALILEO is weaker / needs to improve

- If GALILEO does not explicitly test for “agreeing with the user” failure modes in multimodal settings, this paper suggests a missing evaluation slice.
- If GALILEO relies heavily on instruction-following, it may be vulnerable to the same shortcut: treating user text as more reliable than ambiguous visual evidence.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add an **opinion-injection sycophancy eval** for GALILEO (one-round + two-round), ideally with both misleading and corrective follow-ups.
- [ ] Measure a **modality gap** analog: compare image input vs “gold text description” (or a strong captioner) to see if sycophancy is primarily due to visual uncertainty.
- [ ] Run a **visual confidence sweep** (resolution, noise, blur) and check whether sycophancy tracks confidence.
- [ ] Consider a lightweight **instruction reliability classifier** or a reflection step that decides whether a user follow-up is corrective vs misleading before updating the answer.

## Quotes / details to potentially cite

- They name the phenomenon **“sycophantic modality gap”**: sycophancy is “significantly more prominent when MLLMs process image inputs” than equivalent text.
- Their interpretation: the common **pipelined training paradigm** (large text pretraining + much smaller multimodal alignment) yields lower confidence on images, increasing susceptibility.
- Their observed trade-off: naive SFT to resist misleading inputs can make the model **“overly resistant to corrective instructions (stubborn even if it is wrong)”**.
- SRT’s 3-stage structure: **image textualization → reflection → summarization/conclusion** to decide whether to comply with the user instruction.