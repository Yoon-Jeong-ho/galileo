# Benchmarking and Mitigating Sycophancy in Medical Vision Language Models

- Year: 2025
- Venue: arXiv
- Authors: Zikun Guo, Jingwei Lv, Xinyue Xu, Shu Yang, Jun Wen, Di Wang, Lijie Hu
- URL: https://arxiv.org/abs/2509.21979
- BibTeX key (if we add it): guo2025medical_vlm_sycophancy_viper
- Tags: sycophancy, medical, vision-language-models, robustness, benchmark, prompt-mitigation

## One-sentence takeaway

Medical VLMs frequently “flip” correct image-grounded answers when exposed to social/authority cues, and a simple two-stage prompt that filters non-evidentiary cues (VIPER) measurably reduces these sycophantic failures.

## What problem does it solve?

- Quantifies a safety-critical failure mode (sycophancy) for medical VLMs under realistic social pressure, where user-provided cues conflict with visual evidence.
- Provides a benchmark + taxonomy so we can compare models/mitigations beyond vanilla VQA accuracy.

## What is the core method / protocol?

- Construct a **medical sycophancy benchmark** by taking medical VQA items and creating *paired prompts*:
  - **Baseline/neutral** question.
  - **Pressured** version(s) of the same question with injected social cues.
- Uses a **hierarchical medical VQA** framing (multiple-choice), drawing items from PathVQA, SLAKE, and VQA-RAD, and stratifying by organ system / modality / question type (paper claims 5,000 items).
- Defines sycophancy behaviorally as **flip rate**: when a model that was correct under neutral conditions becomes wrong under pressure.
- Evaluates **16 VLMs** (open-source, commercial, and medical-specialist).
- Mitigation: **VIPER (Visual Information Purification for Evidence-based Responses)**
  - A **single-call, two-stage prompting** strategy:
    1) *Content filter* removes/ignores non-evidentiary social cues.
    2) *Medical expert* response constrained to evidence-first reasoning/output.
  - Paper also analyzes attention shifts (image evidence tokens vs “social” tokens) to support a mechanistic story.

## What are the key metrics?

- Baseline VQA accuracy (neutral prompt).
- **Sycophancy / flip rate** under each social-pressure template.
- “Resistance” after mitigation (reported as a %; higher is better).
- Secondary slices: by pressure type, question type, modality/organ system.

## What are the main results?

- Sycophancy is widespread: paper reports that across models/pressure types, **~40–75% of initially-correct answers flip** under at least one social pressure.
- Vulnerability depends strongly on pressure type; they highlight triggers like **mimicry**, **expert correction/authority**, and **technological self-doubt**.
- Baseline accuracy / model size is only weakly predictive of sycophancy (i.e., strong models still fail under social pressure).
- VIPER reduces sycophancy without obviously trading off baseline accuracy; paper reports **average resistance ~40.6%** and **up to 94.7%** for the best-case model.

## How is this similar to GALILEO?

- Focuses on *robustness to interaction-context perturbations* (here: social/authority cues) rather than only task competence.
- Uses a clean paired-evaluation design (neutral vs perturbed prompt) that isolates a specific failure mechanism.

## How is this different from GALILEO?

- Domain is medical VQA (multimodal) with multiple-choice structure; GALILEO may target different domains/tasks or more general agentic settings.
- Mitigation is primarily prompt-structural (filter + constrained answering), not training-time alignment.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO has training-time controls or guarantees (vs prompt-only), it can claim robustness beyond prompt engineering.
- If GALILEO evaluates multi-step/agentic decisions, it may cover higher-stakes failure propagation than single-turn VQA.

## Where GALILEO is weaker / needs to improve

- If GALILEO does not explicitly test “social pressure” conditions, it may miss a major class of real-world interaction failures.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a **sycophancy-style robustness slice**: paired prompts where a user/agent message injects authority/consensus/emotional pressure that conflicts with evidence.
- [ ] Report **flip-rate** (correct→incorrect) as a primary safety metric alongside accuracy.
- [ ] Try a VIPER-like ablation: **(a)** filter non-evidence cues **(b)** force evidence-first response format; compare to naive “be careful” prompting.
- [ ] If we have attention/logit attribution tooling, add a mechanistic analysis: does the model attend more to “social” tokens under pressure?

## Quotes / details to potentially cite

- Abstract-level claim: “perceived authority and user mimicry are powerful triggers”.
- Benchmark: “5,000 multiple choice VQA items … augmented with seven … social pressure templates” (from the paper’s intro section).
- VIPER: “two stage … Content Filter … followed by a Medical Expert phase …” (intro/figure caption).
