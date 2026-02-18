# Towards LLMs Robustness to Changes in Prompt Format Styles

- Year: 2025
- Venue: NAACL SRW 2025 (arXiv)
- Authors: Lilian Ngweta; Kiran Kate; Jason Tsay; Yara Rizk
- URL: https://arxiv.org/abs/2504.06969
- BibTeX key (if we add it): ngweta2025mof
- Tags: prompt-robustness, format, style, perturbations, few-shot

## One-sentence takeaway

Mixture of Formats (MOF) reduces *style-induced* prompt brittleness by deliberately mixing multiple formatting styles across few-shot exemplars (and asking the model to rewrite exemplars in different styles), improving worst-case vs format changes.

## What problem does it solve?

- LLM performance can swing significantly under *non-semantic* prompt-format changes ("prompt brittleness").
- This work focuses on **style-induced** brittleness: changing surface formatting conventions in few-shot prompts (e.g., different field labels / separators / layout) changes accuracy.

## What is the core method / protocol?

- **MOF prompting** (for few-shot prompts):
  - Present each few-shot example using a **distinct format style** (e.g., different field markers / layouts).
  - Additionally, instruct the model to **rewrite each example in a different style** (so the model is exposed to multiple styles rather than overfitting to one).
- Motivation/analogy: style diversification to prevent spurious correlation between “style” and label, inspired by style-diverse training ideas in vision.
- Evaluation setup (as described):
  - Use multiple tasks/datasets drawn from **SuperNaturalInstructions**.
  - Compare **traditional few-shot prompts** vs the same prompts converted into MOF.
  - Use a “format style generator” (they cite FormatSpread) to produce style variants.

## What are the key metrics?

- **Spread** of performance across prompt styles: max accuracy − min accuracy over different formatting styles (used as a proxy for brittleness).
- Accuracy across tasks under prompt-format perturbations.

## What are the main results?

- MOF generally **reduces performance spread** (i.e., reduces brittleness) relative to traditional prompts across many datasets/models.
- MOF performance is often comparable to or better than traditional prompts, though there are cases where traditional prompts do better.
- Reported across several open LLMs (examples shown include Llama-3-70B-Instruct, Llama-2-13B(-chat), Falcon-11B).

## How is this similar to GALILEO?

- Common theme: **robustness under sequential/interactional perturbations**.
- MOF is essentially a *control/augmentation* trick aimed at stabilizing behavior under superficial changes—conceptually adjacent to robustness protocols where we vary the “wrapper” around the same underlying task.

## How is this different from GALILEO?

- MOF targets **single-prompt / few-shot formatting variation**, not multi-turn conversational drift/pressure.
- The instability metric is **style-induced accuracy spread**, not turn-by-turn failure time, recovery, or trajectory-level robustness.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO is about multi-turn dynamics, it can contribute **longitudinal protocols** (turn-of-failure, recovery-to-truth) that go beyond single-prompt format sensitivity.

## Where GALILEO is weaker / needs to improve

- GALILEO writeup/protocols may under-emphasize **format/style brittleness** as a confounder (changes in wrappers can masquerade as “drift”).

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a small “format robustness” appendix experiment: take a fixed multi-turn scenario and perturb surface formatting of the dialogue turns (speaker tags, delimiter style, field labels) to quantify sensitivity.
- [ ] In methods section, explicitly note that we hold formatting constant (or intentionally vary it) to avoid conflating drift with wrapper changes.
- [ ] Consider MOF-like **style diversification** for any few-shot / rubric-based judge prompts used inside GALILEO pipelines.

## Quotes / details to potentially cite

- "LLMs ... are sensitive to non-semantic changes in prompt formats" (problem framing).
- MOF: "diversifying the styles used in the prompt few-shot examples" to reduce style-induced brittleness.
- Metric: spread = "difference between the best performing prompt (maximum accuracy) and the worst performing prompt (minimum accuracy)" across styles.
