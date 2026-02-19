# PsychiatryBench: A Multi-Task Benchmark for LLMs in Psychiatry

- Year: 2025
- Venue: arXiv
- Authors: Aya E. Fouda; Abdelrahamn A. Hassan; Radwa J. Hanafy; Mohammed E. Fouda
- URL: https://arxiv.org/abs/2509.09711
- BibTeX key (if we add it): psychiatrybench2025fouda
- Tags: domain-benchmark, psychiatry, multi-task, multi-turn, safety, consistency, llm-as-judge

## One-sentence takeaway

A psychiatry-specific benchmark derived from expert textbooks/casebooks with 11 QA-style task formats, showing frontier LLMs still exhibit major consistency/safety gaps on multi-turn follow-up and management tasks.

## What problem does it solve?

- Existing mental-health/psychiatry LLM evaluations often rely on small interview corpora, social media posts, or synthetic dialogues that (per authors) have limited clinical validity and do not stress real diagnostic/treatment/management reasoning.
- Need a more clinically grounded, high-stakes evaluation that better reflects psychiatric decision-making complexity.

## What is the core method / protocol?

- Build *PsychiatryBench*: 5,188 expert-annotated items curated “exclusively” from authoritative psychiatric textbooks and casebooks (e.g., DSM-5-TR clinical cases; psychopharmacology references).
- Define 11 task types spanning (authors’ grouping):
  - diagnosis/classification (diagnostic reasoning)
  - treatment decisions + treatment follow-up (including multi-turn follow-up)
  - management plan (care coordination / risk / follow-up)
  - clinical approach
  - sequential case analysis (evolving info across time)
  - foundational knowledge QA + MCQ + extended matching items (EMI)
- Evaluate multiple frontier LLMs and some medical open models; scoring uses conventional metrics plus an “LLM-as-judge” similarity scoring framework.

## What are the key metrics?

- Conventional QA metrics (not fully detailed in the abstract; likely exact match / F1 / accuracy depending on format).
- “LLM-as-judge” similarity scoring (grading model answers vs reference via another model), used as an auxiliary/alternative to strict matching.
- Safety/consistency analyses emphasized for multi-turn follow-up and management-style tasks.

## What are the main results?

- Even strong frontier models show “substantial gaps” in clinical consistency and safety.
- Weakest areas highlighted: multi-turn follow-up and management planning tasks (i.e., settings where the model must stay coherent and clinically appropriate across steps/time).

## How is this similar to GALILEO?

- Shared emphasis on *multi-turn failure modes*: consistency degradation across turns and unsafe/incorrect behavior in longitudinal interaction.
- Benchmark philosophy: stress test *reliability* (not just single-turn accuracy) and highlight gaps that are masked by simpler evaluations.

## How is this different from GALILEO?

- Domain-locked to adult psychiatry; tasks largely framed as clinical QA and case-based reasoning rather than general agentic behavior in open-world tasks.
- Uses curated textbook/casebook sources (stronger grounding) but still not a direct agent setting.
- Uses LLM-as-judge similarity scoring, which may introduce evaluator/model bias and can blur fine-grained failure categorization.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO focuses on domain-general multi-turn instability with explicit controls (drift vs evidence-driven revision) and robust metrics (time-to-failure, survival-style), it may provide a cleaner causal/protocol story than a broad domain benchmark with heterogeneous task formats.

## Where GALILEO is weaker / needs to improve

- GALILEO may lack domain-grounded “gold” references and expert-validated, high-stakes content analogous to psychiatry casebooks; could benefit from borrowing the *authoritative-source curation* principle.

## Action items for GALILEO (experiments / method / writing)

- [ ] In related work, cite PsychiatryBench as evidence that even frontier models struggle with *multi-turn follow-up* and *management planning* consistency/safety in a high-stakes domain.
- [ ] Consider a GALILEO appendix note: contrast strict-match metrics vs LLM-as-judge similarity scoring; discuss how evaluator choice affects measured drift/instability.
- [ ] Steal framing language: explicitly define “high-stakes” and enumerate error consequences (misdiagnosis, unsafe recommendations) as an analogy for high-stakes agent settings.

## Quotes / details to potentially cite

- “PsychiatryBench comprises eleven distinct question-answering tasks … totaling 5,188 expert-annotated items.”
- “Our results reveal substantial gaps in clinical consistency and safety, particularly in multi-turn follow-up and management tasks …”
