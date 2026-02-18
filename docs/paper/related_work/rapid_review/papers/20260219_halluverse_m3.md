# Halluverse-M^3: A multitask multilingual benchmark for hallucination in LLMs

- Year: 2026
- Venue: arXiv (cs.CL, cs.AI)
- Authors: Parichit Sharma; Erchin Serpedin; Hasan Kurban
- URL: https://arxiv.org/abs/2602.06920
- BibTeX key (if we add it): HalluverseM3Sharma2026
- Tags: hallucination, multilingual, benchmark, detection, summarization, QA

## One-sentence takeaway

HalluVerse-M3 is a controlled-edit, human-validated benchmark for fine-grained hallucination detection across 4 languages and 2 generation tasks, showing strong task/language effects (QA easier than dialogue summarization; English best, Hindi worst; sentence-level hardest).

## What problem does it solve?

- Lack of a *single* benchmark that simultaneously covers (a) multiple languages, (b) multiple generation tasks, and (c) fine-grained hallucination categories, to study how hallucination/detection varies across these axes.

## What is the core method / protocol?

- Build a dataset with paired (reference, hallucinated) generations via a *controlled editing* procedure.
- Languages: English, Arabic, Hindi, Turkish.
- Tasks: question answering (QA) and dialogue summarization (DS).
- Hallucination categories: entity-level, relation-level, sentence-level.
- Human validation to ensure the hallucinated edits are correctly aligned to the intended category and the reference.
- Evaluate a mix of open-source + proprietary LLMs on *hallucination detection* in this fine-grained setting.

## What are the key metrics?

- Hallucination detection accuracy (reported per task, language, and hallucination type).

## What are the main results?

- QA is consistently easier than dialogue summarization for hallucination detection.
- Sentence-level hallucinations remain challenging even for the strongest evaluated models.
- Best performance in English; performance degrades in lower-resource languages; Hindi is reported as the lowest detection accuracy.

## How is this similar to GALILEO?

- Provides an evaluation resource for hallucination behavior and detection, with an explicit taxonomy (entity/relation/sentence) that could be mapped to GALILEO’s error categories.
- Highlights *distribution shift across languages/tasks* as a core factor in hallucination-related reliability.

## How is this different from GALILEO?

- Focus is a benchmark/dataset for hallucination *detection*, not a broader reliability protocol (e.g., longitudinal instability, drift/revision separation, or multi-turn survival-style evaluation).
- Hallucinations are created via controlled edits relative to references, rather than naturally occurring long-horizon conversational degradation.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO targets temporal/multi-turn dynamics, it captures failure modes (drift, recovery, intervention effects) not addressed by static paired examples.

## Where GALILEO is weaker / needs to improve

- If GALILEO lacks multilingual coverage or fine-grained hallucination-type annotations, HalluVerse-M3 is a clear neighbor that can motivate adding these axes or providing a mapping.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider a GALILEO appendix/baseline that reports results split by hallucination granularity (entity vs relation vs sentence) and language, or at least discuss how these factors can confound reliability claims.
- [ ] Cite HalluVerse-M3 as evidence that sentence-level (global) hallucinations are systematically harder than localized entity/relation edits.

## Quotes / details to potentially cite

- Dataset axes: 4 languages (English/Arabic/Hindi/Turkish), 2 tasks (QA, dialogue summarization), 3 hallucination types (entity/relation/sentence).
- Main finding summary: QA easier than dialogue summarization; sentence-level hardest; English highest, Hindi lowest detection accuracy.
- Resource: https://huggingface.co/datasets/sabdalja/HalluVerse-M3
