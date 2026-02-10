# ReviseQA (2024) — Notes

## What it is
- Task setting: models revise an answer after receiving feedback; evaluates when revisions improve vs degrade.
- Key angle for us: revision dynamics under *feedback pressure* can cause truth to be abandoned.

## How it relates to GALILEO
- We differ in protocol scope: GALILEO measures **multi-round** dynamics (Survival / TOF / Recovery) on **ground-truth** tasks with explicitly designed pressure personas.
- We add a **Neutral Re-asking Control** as a drift baseline to separate persona mechanisms from generic multi-turn variance.
- Our recovery metric is measured **conditional on flip**, explicitly decoupling “stayed correct” from “returned to truth”.

## How to cite / position
- Use as prior evidence that “revision under feedback can be harmful”, motivating why we need multi-turn dynamics rather than single-turn accuracy.

## BibTeX
- TODO: add the BibTeX entry for ReviseQA to `references.bib` (verify exact title/authors/venue).
