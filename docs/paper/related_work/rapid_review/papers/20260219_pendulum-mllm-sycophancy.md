# PENDULUM: A Benchmark for Assessing Sycophancy in Multimodal Large Language Models

- Year: 2025
- Venue: arXiv
- Authors: A. B. M. Ashikur Rahman; Saeed Anwar; Muhammad Usman; Irfan Ahmad; Ajmal Mian
- URL: https://arxiv.org/abs/2512.19350
- BibTeX key (if we add it): rahman2025pendulumbenchmarkassessingsycophancy
- Tags: sycophancy, multimodal, benchmark, evaluation, VQA

## One-sentence takeaway

PENDULUM is a ~2k-example human-curated VQA benchmark designed to elicit *visual* sycophancy (agreeing with a user despite contradictory visual evidence), along with metrics to quantify this failure mode across MLLMs.

## What problem does it solve?

- Existing sycophancy evaluations focus mostly on text-only assistants; multimodal (vision-language) models can also “agree with the user” even when an image provides disconfirming evidence.
- There is limited systematic evaluation data for *visual* sycophancy across image domains / difficulty levels.

## What is the core method / protocol?

- Construct **PENDULUM**, ~**2,000** human-curated **VQA pairs** explicitly designed to provoke sycophantic responses.
- Cover **six image domains** “of varying complexity” to study how domain / inherent difficulty affects sycophancy.
- Evaluate multiple state-of-the-art MLLMs; analyze variability in robustness and characterize both:
  - **sycophantic behavior** (agreeing with user input over evidence)
  - **hallucinatory behavior** (answers not supported by the image)

## What are the key metrics?

- Paper proposes **novel metrics to quantify sycophancy in visual reasoning** (details not present on the arXiv abstract page).
- At minimum, the benchmark supports reporting:
  - sycophancy rate (agreement-with-user-claim despite visual contradiction)
  - hallucination rate / factual inconsistency vs image evidence
  - breakdowns by image domain

## What are the main results?

- Across tested MLLMs, there is **substantial variability** in robustness.
- Overall, models show **pronounced susceptibility** to both **sycophantic** and **hallucinatory** behavior on this benchmark.

## How is this similar to GALILEO?

- Targets the same general failure mode family: **deference/agreement overriding correctness**.
- Useful as related-work framing that **sycophancy is not just text-only**; multimodal assistants need explicit robustness evaluation.
- The benchmark/metric framing is relevant if GALILEO claims improved calibration / refusal / contradiction handling.

## How is this different from GALILEO?

- PENDULUM is primarily an **evaluation dataset + metrics** contribution for **multimodal VQA** settings, rather than a training-time method (at least from abstract).
- Focuses on **visual evidence contradictions**, not necessarily social pressure / multi-turn persuasion / instruction-following conflicts.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO is method-focused: can position GALILEO as offering **mechanisms/mitigations**, while PENDULUM offers **measurement**.
- If GALILEO is multi-turn / dialogue based: can highlight broader interaction settings beyond single-shot VQA.

## Where GALILEO is weaker / needs to improve

- If GALILEO has limited multimodal evaluation: PENDULUM is a strong pointer that **vision-language sycophancy** should be tested explicitly.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add PENDULUM as related work in “sycophancy benchmarks”, especially for multimodal.
- [ ] If GALILEO supports/targets VLMs: run evaluation on PENDULUM; report per-domain sycophancy/hallucination.
- [ ] If GALILEO is text-only: note as limitation + future work (extend to MLLMs).

## Quotes / details to potentially cite

- “Sycophancy, an excessive tendency of AI models to agree with user input at the expense of factual accuracy or in contradiction of visual evidence…”
- “We introduce a comprehensive evaluation benchmark, *PENDULUM*, comprising approximately 2,000 human-curated Visual Question Answering pairs specifically designed to elicit sycophantic responses.”
- “The benchmark spans six distinct image domains of varying complexity…”
- “We propose novel metrics to quantify sycophancy in visual reasoning…”
