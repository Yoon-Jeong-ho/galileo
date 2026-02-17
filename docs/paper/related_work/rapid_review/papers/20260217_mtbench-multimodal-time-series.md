# MTBench: A Multimodal Time Series Benchmark for Temporal Reasoning and Question Answering

- Year: 2025
- Venue: arXiv
- Authors: Jialin Chen; Aosong Feng; Ziyu Zhao; Juan Garza; Gaukhar Nurbek; Cheng Qin; Ali Maatouk; Leandros Tassiulas; Yifeng Gao; Rex Ying
- URL: https://arxiv.org/abs/2503.16858
- BibTeX key (if we add it): mtbench_chen_2025
- Tags: benchmark, temporal-reasoning, multimodal, time-series, QA

## One-sentence takeaway

MTBench pairs real time series (finance + weather) with aligned text (news + reports) to test whether LLMs can jointly reason over narrative context and numerical temporal dynamics for forecasting and QA.

## What problem does it solve?

- Existing multimodal time-series datasets often test each modality in isolation and don’t stress *cross-modal reasoning* (e.g., linking a textual event to subsequent time-series movement) or question answering that requires integrating both.
- Need a benchmark that surfaces failure modes like weak long-term temporal dependency handling and poor causal interpretation in multimodal settings.

## What is the core method / protocol?

- Construct a benchmark of paired (time series, text) examples in two domains:
  - Finance: financial news aligned with corresponding stock price movements.
  - Weather: weather reports aligned with historical temperature records.
- Define a set of tasks requiring joint reasoning:
  - Time-series forecasting.
  - Semantic + technical trend analysis.
  - News-driven question answering.
- Evaluate multiple SOTA LLMs and analyze typical errors (temporal dependency, causal interpretation, multimodal fusion).

## What are the key metrics?

- Not specified in the abstract; likely includes:
  - Forecasting error (e.g., MAE/MSE) on time-series prediction tasks.
  - Accuracy / F1 / exact match for QA.
  - Possibly rubric-based scoring for trend analysis.

## What are the main results?

- SOTA LLMs still struggle with:
  - Capturing long-term dependencies in time series.
  - Interpreting causality in financial and weather trends.
  - Effectively fusing multimodal (text + numeric time-series) signals.

## How is this similar to GALILEO?

- Both are about *evaluation protocols* that expose brittleness/limitations of LLM reasoning under more realistic, structured settings.
- MTBench’s emphasis on “temporal dependencies” connects loosely to GALILEO’s concern with degradation across interactions/time (even though MTBench is not multi-turn dialogue).

## How is this different from GALILEO?

- MTBench focuses on multimodal time-series + text integration (finance/weather), not multi-turn conversational robustness, sycophancy, or social-pressure dynamics.
- GALILEO’s core is interaction/trajectory robustness (turn-level failures, drift), whereas MTBench is a dataset-style benchmark for temporal reasoning and QA.

## Where GALILEO is stronger / cleaner (if true)

- Clearer linkage to *multi-turn interaction dynamics* and robustness-to-manipulation settings (GALILEO’s focus).
- Metrics like time-to-failure/turn-of-failure are more directly aligned with conversational degradation than typical dataset metrics.

## Where GALILEO is weaker / needs to improve

- Less coverage of multimodal temporal reasoning; MTBench suggests a direction for broader “temporal reasoning under external evidence streams” evaluations.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider a short related-work paragraph positioning: “temporal dependency + fusion failures show up outside dialogue too; our work targets analogous degradation phenomena in multi-turn interaction.”
- [ ] (Optional) If we ever add multimodal variants, MTBench is a candidate reference for paired (text, time-series) construction.

## Quotes / details to potentially cite

- “MTbench comprises paired time series and textual data, including financial news with corresponding stock price movements and weather reports aligned with historical temperature records.”
- “Tasks … including time-series forecasting, semantic and technical trend analysis, and news-driven question answering (QA).”
- “Findings reveal significant challenges … difficulties in capturing long-term dependencies … interpreting causality … and effectively fusing multimodal information.”
