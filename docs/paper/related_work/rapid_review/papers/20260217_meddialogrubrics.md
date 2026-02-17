# MedDialogRubrics: A Comprehensive Benchmark and Evaluation Framework for Multi-turn Medical Consultations in Large Language Models

- Year: 2026
- Venue: arXiv
- Authors: Lecheng Gong, Weimin Fang, Ting Yang, Dongjie Tao, Chunxiao Guo, Peng Wei, Bo Xie, Jinqun Guan, Zixiao Chen, Fang Shi, Jinjie Gu, Junwei Liu
- URL: https://arxiv.org/abs/2601.03023
- BibTeX key (if we add it): meddialogrubrics_gong_2026
- Tags: multi-turn, medical, benchmark, evaluation, rubrics, diagnostic-reasoning, information-gathering

## One-sentence takeaway
A synthetic multi-turn medical consultation benchmark (5.2k cases) with fine-grained, expert-refined rubric items to score whether models ask the “must-ask” diagnostic questions and follow plausible diagnostic reasoning.

## What problem does it solve?
- Existing medical dialogue benchmarks/frameworks don’t rigorously evaluate *multi-turn* information gathering + diagnostic reasoning with fine-grained, case-specific criteria.
- Real EHR-based evaluations raise privacy/data-governance issues.

## What is the core method / protocol?
- Construct **5,200 synthetic patient cases** via a **multi-agent synthesis pipeline** that generates patient records + chief complaints from disease knowledge (explicitly avoiding real EHRs).
- A **Patient Agent** is constrained to a set of **atomic medical facts** and uses a **dynamic guidance mechanism** to detect/correct hallucinations during the dialogue to keep simulated cases coherent/plausible.
- Generate **>60k evaluation rubrics** with an LLM+expert pipeline:
  - retrieve Evidence-Based Medicine (EBM) guidelines,
  - use reject sampling to produce a prioritized set of rubric items (notably “must-ask” items) per case,
  - clinical experts refine the rubrics.
- Evaluate SOTA models across multiple assessment dimensions (paper claims current models struggle).

## What are the key metrics?
- Rubric-based multi-dimensional scoring of consultation quality; emphasis on:
  - coverage of prioritized “must-ask” questions (information gathering)
  - diagnostic reasoning quality / coherence (as operationalized by rubric items)
  - plausibility/consistency of the simulated patient side (as a benchmark property)

## What are the main results?
- Across multiple dimensions, current models show “substantial challenges” on multi-turn diagnostic dialogue.
- The authors argue progress will require **dialogue management architecture** improvements, not just incremental base-model tuning.

## How is this similar to GALILEO?
- Shared theme: **multi-turn evaluation protocols** that try to be more diagnostic than a single scalar accuracy.
- Uses fine-grained, structured criteria (rubrics) somewhat analogous in spirit to decomposing failures over turns.

## How is this different from GALILEO?
- Domain: clinical multi-turn diagnosis vs GALILEO’s focus on robustness/instability under multi-turn perturbations.
- Primary endpoint is rubric completion/quality rather than time-to-failure / survival-style robustness.
- Heavy emphasis on benchmark *construction* (synthetic case generation + expert rubric curation).

## Where GALILEO is stronger / cleaner (if true)
- If GALILEO targets general robustness, it may provide more model-agnostic, domain-agnostic measures (vs domain-specific medical rubrics).

## Where GALILEO is weaker / needs to improve
- Case-specific **prioritized rubric items** (“must-ask”) provide a very interpretable failure signal; GALILEO could benefit from similarly interpretable, per-turn “expected invariants” in its own domains.

## Action items for GALILEO (experiments / method / writing)
- [ ] Consider adding a *rubric-like* layer: for each multi-turn scenario, define a small set of “must-maintain/must-ask” invariants to diagnose failures beyond a single survival curve.
- [ ] In related work, cite as an example of **fine-grained rubric-based multi-turn evaluation** and synthetic-data governance approach.

## Quotes / details to potentially cite
- “MedDialogRubrics, a novel benchmark comprising 5,200 synthetically constructed patient cases and over 60,000 fine-grained evaluation rubrics … refined by clinical experts … to assess the multi-turn diagnostic capabilities of LLM.”
- “We propose … rubric-generation pipeline that retrieves Evidence-Based Medicine (EBM) guidelines and utilizes … reject sampling to derive a prioritized set of rubric items (‘must-ask’ items) for each case.”
- “Improving medical dialogue will require advances in dialogue management architectures, not just incremental tuning of the base-model.”
