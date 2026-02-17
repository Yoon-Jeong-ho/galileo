# A Survey on Recent Advances in LLM-Based Multi-turn Dialogue Systems

- Year: 2024
- Venue: ACM Computing Surveys (preprint on arXiv)
- Authors: (not captured in this rapid run; see arXiv page)
- URL: https://arxiv.org/abs/2402.18013
- BibTeX key (if we add it): survey-llm-mt-dialogue-2024-yi
- Tags: survey, multi-turn, dialogue-systems, LLMs, datasets, evaluation

## One-sentence takeaway

A broad survey of LLM-based open-domain and task-oriented multi-turn dialogue systems, emphasizing adaptation methods, datasets, and evaluation—useful background but largely not a “closest-neighbor” protocol paper.

## What problem does it solve?

- Consolidates a fast-moving literature on multi-turn dialogue systems (ODD + TOD) in the LLM era.
- Organizes methods for adapting LLMs to dialogue tasks and summarizes common datasets/metrics.

## What is the core method / protocol?

- Survey taxonomy rather than a new experimental protocol.
- Frames multi-turn dialogue as sequence-to-sequence over turns (user utterances + history → responses) and discusses challenges like long-range dependency and cross-turn consistency.

## What are the key metrics?

- As a survey: covers common dialogue evaluation families (automatic metrics + human evaluation), and task success style metrics for TOD.
- (No single new metric introduced in the abstract/intro excerpts reviewed here.)

## What are the main results?

- Narrative synthesis: summarizes LLMs and adaptation approaches; surveys recent ODD/TOD advances; highlights open problems/future directions.

## How is this similar to GALILEO?

- Overlaps in the *multi-turn* setting and concerns about context maintenance / consistency across turns.
- Provides pointers to datasets and evaluation practices that may be referenced in related work.

## How is this different from GALILEO?

- Not focused on *robustness-to-pressure / drift / sycophancy* protocols.
- No “time-to-failure”/survival-style measurement emphasis (at least not in the parts reviewed).

## Where GALILEO is stronger / cleaner (if true)

- GALILEO can offer a more targeted contribution: multi-turn robustness protocols/metrics under adversarial or social pressure, rather than a broad survey.

## Where GALILEO is weaker / needs to improve

- Ensure our related-work framing cleanly situates robustness/drift evaluation within the broader dialogue-system evaluation landscape (which surveys like this summarize).

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider citing this survey for general background on multi-turn dialogue systems (ODD/TOD split, evaluation overview) if we need an “umbrella” citation.
- [ ] Double-check whether the full paper has a section that mentions *consistency/stability* metrics explicitly; if so, cite that section.

## Quotes / details to potentially cite

- Abstract: “comprehensive review of research on multi-turn dialogue systems … focus on … based on large language models (LLMs)” and covers “open-domain dialogue (ODD) and task-oriented dialogue (TOD) … datasets and evaluation metrics.”

---

## Note on duplication in this rapid-review queue

This queue item duplicates an already-written note for the same arXiv id in this repo; see: papers/20260217_survey-llm-multi-turn-dialogue-systems.md
