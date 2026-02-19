# How Do LLMs Fail In Agentic Scenarios? A Qualitative Analysis of Success and Failure Scenarios of Various LLMs in Agentic Simulations

- Year: 2025
- Venue: arXiv
- Authors: (TODO)
- URL: https://arxiv.org/abs/2512.07497
- BibTeX key (if we add it): (TODO)
- Tags: agents, tool-use, failure-analysis, qualitative

## One-sentence takeaway

(TODO after skim) A qualitative taxonomy of LLM failure modes in agentic simulations.

## What problem does it solve?

- Agent benchmarks often report aggregate success rates but under-describe *how* and *why* failures occur.

## What is the core method / protocol?

- (TODO) Run multiple LLMs in agentic environments; analyze traces; produce failure categories.

## Key failure categories (expected)

- (TODO) Planning errors, tool misuse, compounding mistakes, instruction drift, susceptibility to misleading cues.

## How is this similar to GALILEO?

- Trace-based evaluation: both benefit from per-step artifacts and diagnosing *where in the interaction* things go wrong.

## How is this different from GALILEO?

- This is qualitative and environment-dependent; GALILEO offers a *controlled* persona protocol with automatic ground-truth labeling (survival / turn-of-failure / recovery).

## Action items for GALILEO (writing)

- [ ] If their taxonomy includes social influence / deference, cite it to broaden motivation.
- [ ] Use as a contrast: “we focus on one failure axis and make it automatically measurable.”
