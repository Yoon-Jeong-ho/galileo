# A Survey on Recent Advances in LLM-Based Multi-turn Dialogue Systems

- Year: 2024
- Venue: ACM Computing Surveys (preprint on arXiv)
- Authors: Zihao Yi, Jiarui Ouyang, Zhe Xu, Yuwen Liu, Tianhao Liao, Haohao Luo, Ying Shen
- URL: https://arxiv.org/abs/2402.18013
- BibTeX key (if we add it): yi2024survey_multiturn_dialogue
- Tags: survey, multi-turn, dialogue-systems, LLMs, evaluation

## One-sentence takeaway

A broad survey of LLM-based multi-turn dialogue systems (open-domain + task-oriented), covering adaptation methods, datasets/metrics, and open problems—useful for positioning but not a direct metric/protocol neighbor to GALILEO.

## What problem does it solve?

- Consolidates a fast-moving literature on *multi-turn* dialogue systems in the LLM era.
- Organizes approaches for adapting LLMs to dialogue tasks, and summarizes datasets + evaluation methods.

## What is the core method / protocol?

- Survey / taxonomy style paper (no new benchmark).
- Breaks the space into:
  - LLMs + adaptation approaches for downstream dialogue tasks (e.g., prompt-based, fine-tuning, alignment-style methods).
  - LLM-based open-domain dialogue (ODD) vs task-oriented dialogue (TOD) systems.
  - Datasets and evaluation metrics used for multi-turn dialogue.
  - Future directions / challenges (context handling, consistency, safety, evaluation reliability, etc.).

## What are the key metrics?

- As a survey, it catalogs commonly used metrics rather than proposing a new one.
- Likely coverage includes (typical in this area):
  - Automatic metrics (BLEU/ROUGE variants, BERTScore, etc.)
  - Task success / slot-filling metrics for TOD
  - Human evaluation dimensions (coherence, helpfulness, consistency, safety)
  - Multi-turn-specific criteria (context usage, persona/stance consistency, long-horizon memory)

## What are the main results?

- No primary experimental result; the “result” is the structured synthesis of methods, datasets, and evaluation practices.

## How is this similar to GALILEO?

- Overlaps in problem framing: multi-turn interactions introduce *drift*, *consistency* issues, and evaluation pitfalls.
- Useful source for related-work prose on multi-turn dialogue evaluation and challenges (context length, long-range dependencies, consistency across turns).

## How is this different from GALILEO?

- Not focused on *robustness-to-pressure* or *time-to-failure/survival-style* evaluation.
- Does not propose a crisp, reproducible protocol/metric for “when the model breaks” under adversarial or persuasive multi-turn interaction.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO has a concrete multi-turn robustness protocol + quantitative, turn-indexed failure measurement, it will read as a more *operationalized* evaluation contribution than a broad survey.

## Where GALILEO is weaker / needs to improve

- Surveys can cite a wider span of dialogue-system work (ODD/TOD) and standard evaluation practices; GALILEO should ensure it clearly situates itself within those established dialogue evaluation dimensions.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add 1–2 related-work paragraphs positioning GALILEO relative to multi-turn dialogue evaluation (ODD/TOD), emphasizing why standard metrics/human eval are insufficient for robustness-to-pressure / longitudinal failure.
- [ ] Mine this survey’s cited “evaluation metrics” section for canonical citations to anchor our critique of current practice.

## Quotes / details to potentially cite

- “Multi-turn dialogue systems need to maintain context over several turns… challenges, such as handling long-range dependencies, maintaining consistency across turns…” (intro-level framing; paraphrase + cite).
