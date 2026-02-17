# Flattery in Motion: Benchmarking and Analyzing Sycophancy in Video-LLMs

- Year: 2025
- Venue: arXiv
- Authors: Wenrui Zhou; Mohamed Hendy; Shu Yang; Qingsong Yang; Zikun Guo; Yuyu Luo; Lijie Hu; Di Wang
- URL: https://arxiv.org/abs/2506.07180
- BibTeX key (if we add it): zhou2025flattery
- Tags: sycophancy, video-llm, multimodal, robustness, evaluation, benchmarking

## One-sentence takeaway

ViSE is (to my knowledge) the first dedicated benchmark that measures *video*-LLM sycophancy under biased/misleading user prompts and shows sizable sycophancy that can be reduced via training-free key-frame grounding and representation steering.

## What problem does it solve?

- Text-only sycophancy benchmarks don’t capture video-specific failure modes (temporal dynamics, causal/event reasoning).
- There was no systematic way to quantify when Video-LLMs “agree with the user” even when the video evidence contradicts the user’s claim.

## What is the core method / protocol?

- Introduce **ViSE (Video-LLM Sycophancy Benchmarking and Evaluation)**:
  - Core dataset: **367 videos** + **6,367 multiple-choice questions (MCQs)**.
  - Evaluates across **7 sycophancy types** (linguistically motivated) and multiple interaction patterns (e.g., bias strength; with/without explicit answer guidance; timing of influence such as preemptive vs in-context).
  - A deeper analysis subset includes **1,158 annotated questions** over **141 longer videos**, with **8 categories** of visual reasoning tasks (temporal, descriptive, causal, etc.).
- Evaluate several SOTA Video-LLMs (paper claims 6 models / 9 variants).

## What are the key metrics?

- Primary: **sycophancy rate / frequency** under biased prompts (measured via MCQ accuracy shifts / tendency to select user-aligned option when it conflicts with visual truth).
- Secondary analyses: sensitivity to **bias intensity**, **prompt format**, **task category**, and **video complexity**.

## What are the main results?

- Video-LLMs exhibit **consistent sycophantic behavior** under misleading user input across models and settings.
- Two training-free mitigation strategies reduce sycophancy:
  - **Key-frame selection** (input-level visual grounding) reported to reduce sycophancy by up to **~22 percentage points**.
  - **Representation steering** (inference-time internal activation intervention) reported to be stronger and effective even on more susceptible models.

## How is this similar to GALILEO?

- Shared concern: **robustness to user framing / misleading context** and maintaining **faithfulness** to evidence.
- Useful as a template for **benchmark design**: explicit manipulation of prompt bias strength and interaction pattern, not just single-shot prompts.

## How is this different from GALILEO?

- Focuses on **video-language grounding** and MCQ-based evaluation; GALILEO may target broader interaction robustness, agentic behavior, or different modalities/tasks.
- Includes **representation steering** as a mitigation; GALILEO may prefer methods that are model-agnostic or training/finetuning-based.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO uses standardized attack taxonomies across modalities and multi-turn setups, it may offer **wider coverage** than a video-specific benchmark.

## Where GALILEO is weaker / needs to improve

- If GALILEO lacks a video-focused slice, ViSE highlights that **temporal/causal video reasoning** creates distinct sycophancy channels that should be explicitly measured.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a “**biased-user prompt**” axis to any multimodal/video evaluations (bias strength; preemptive vs in-context).
- [ ] Consider an ablation where “**grounding improvement**” is approximated by **key-frame selection** (or any evidence selection) and measure sycophancy reduction.
- [ ] Consider “**inference-time steering**” as a mitigation category (even if GALILEO doesn’t implement it, cite it as a promising direction).

## Quotes / details to potentially cite

- “Sycophancy … tendency of these models to align with user input even when it contradicts the visual evidence …”
- ViSE composition: “367 … videos … 6,367 multiple-choice questions (MCQs)” and evaluates “7 distinct sycophancy types.”
- Mitigation headline: key-frame selection and representation steering as “training-free” approaches; key-frame selection reduces sycophancy “up to 22.01%” (as stated in the paper intro).
