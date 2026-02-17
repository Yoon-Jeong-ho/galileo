# The Chameleon Nature of LLMs: Quantifying Multi-Turn Stance Instability in Search-Enabled Language Models

- Year: 2025
- Venue: NeurIPS 2025 Workshop (MTI-LLM)
- Authors: Shivam Ratnakar*, Sanjay Raghavendra*
- URL: https://arxiv.org/abs/2510.16712
- BibTeX key (if we add it): ratnakar2025chameleon
- Tags: multi-turn, stance-instability, search-enabled, robustness, consistency, evaluation, retrieval

## One-sentence takeaway

Search/RAG-enabled LLMs often “flip” stances across contradictory multi-turn questioning, and the paper proposes a dataset + stance-instability metrics showing this is systematic (not just sampling noise).

## What problem does it solve?

- Quantifies a specific multi-turn failure mode: *stance instability* (“chameleon behavior”) when users ask contradictory/probing follow-ups.
- Targets *search-enabled* LLM settings (LLM + web search / retrieval), where one might expect grounding to reduce inconsistency.

## What is the core method / protocol?

- Build **Chameleon Benchmark Dataset**:
  - 12 controversial domains.
  - 1,180 multi-turn conversations.
  - 17,770 question–answer pairs.
  - Each conversation includes a sequence of “probes” (contradictory questions / evidence requests / tradeoff analyses) designed to test whether the model maintains a coherent stance.
- Evaluate multiple frontier systems (reported in abstract): Llama-4-Maverick, GPT-4o-mini, Gemini-2.5-Flash.
- Stance is analyzed by a (fixed) judge pipeline (per paper figures/description) and aggregated into metrics.

## What are the key metrics?

- **Chameleon Score (0–1):** quantifies stance instability across the conversation (paper describes RMS-style aggregation, also incorporating confidence / contradiction behavior).
- **Source Re-use Rate (0–1):** measures how much the system reuses the same sources (proxy for knowledge diversity in retrieval).

## What are the main results?

- All tested models show substantial stance instability (abstract reports Chameleon Scores ~0.391–0.511; GPT-4o-mini worst).
- Temperature variance is very small (<0.004), argued to imply the effect is not primarily a sampling artifact.
- Reported correlations (abstract):
  - Source re-use vs confidence: r = 0.627 (p < 0.05)
  - Source re-use vs stance changes: r = 0.429 (p < 0.05)
  - Interpretation: limited retrieval diversity encourages pathological deference to query framing.

## How is this similar to GALILEO?

- Same general phenomenon class: **multi-turn robustness / drift / flip behavior** under conversational pressure.
- Shares an evaluation-first framing: define protocol + metric to capture a dynamic failure mode across turns.
- Mechanistic angle (retrieval/source diversity ↔ instability) is adjacent to “why models drift / comply” analyses.

## How is this different from GALILEO?

- Focuses on **stance stability** (often subjective/controversial topics) rather than truthfulness under social pressure, deception, or explicit sycophancy to user beliefs.
- Emphasizes **search-enabled** systems and retrieval/source reuse; GALILEO is likely model- / interaction-protocol-centric rather than RAG-pipeline-centric.
- Workshop paper; may have less comprehensive ablations/controls than a full benchmark release.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO cleanly separates *evidence-driven belief revision* from *socially-driven drift*, that would complement (and possibly improve on) this paper’s stance-flip focus.
- If GALILEO evaluates both with/without retrieval and uses controlled perturbations, it can generalize beyond search-enabled setups.

## Where GALILEO is weaker / needs to improve

- Consider adding a **stance-instability slice** (especially for “controversial domains”), because it captures a user-facing failure mode that is easy to communicate.
- Consider adding a **retrieval diversity / evidence diversity** factor (even if GALILEO is not about RAG) to test whether instability is mediated by “evidence breadth”.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a small protocol inspired by “probing contradictory follow-ups” and report a flip/stability metric across turns.
- [ ] Add an ablation: fixed evidence set vs diverse evidence set (or “repeat same evidence” vs “new evidence”) to test whether diversity reduces drift/flip.
- [ ] In related work: cite this as a *search-enabled multi-turn stance consistency* benchmark + metric proposal.

## Quotes / details to potentially cite

- Dataset scale (abstract): “17,770 … question-answer pairs across 1,180 multi-turn conversations spanning 12 controversial domains.”
- Metrics (abstract): “Chameleon Score (0–1)” and “Source Re-use Rate (0–1)”.
- Temperature-independence claim (abstract): “across-temperature variance (<0.004) suggests the effect is not a sampling artifact.”
