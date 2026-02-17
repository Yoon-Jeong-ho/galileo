# Evaluating the Sensitivity of LLMs to Prior Context

- Year: 2025
- Venue: arXiv
- Authors: Robert Hankache; Kingsley Nketia Acheampong; Liang Song; Marek Brynda; Raad Khraishi; Greig A. Cowan
- URL: https://arxiv.org/abs/2506.00069
- BibTeX key (if we add it): hankache2025sensitivity
- Tags: multi-turn, context-sensitivity, robustness, evaluation, gpqa

## One-sentence takeaway

Multi-turn “prior context” (especially long or domain-shifted) can severely degrade MCQ accuracy on GPQA-derived queries, and simply repositioning the task instruction within context can substantially recover performance.

## What problem does it solve?

- Standard single-turn QA benchmarks (e.g., GPQA) don’t characterize how accuracy changes once models operate in realistic multi-turn settings with accumulating prior turns.
- We need controlled protocols to *vary the amount and kind* of preceding context and quantify sensitivity / degradation.

## What is the core method / protocol?

- Construct benchmarks where each evaluation instance contains:
  - optional prior context / conversation history
  - a target query (multiple-choice, drawn from GPQA Diamond STEM subsets)
  - 4 answer options (1 correct)
- Systematically vary:
  - volume/length of prior context (depth of conversation history)
  - nature of prior context (e.g., domain/content shifts vs domain-consistent history)
  - placement of the task description/instructions inside the context (as a mitigation knob)
- Evaluate multiple closed LLM families (GPT, Claude, Gemini) under these controlled contextual variations.

## What are the key metrics?

- Accuracy on the target multiple-choice question (correct option selected).
- Reported as absolute accuracy and relative drop vs a no-prior-context baseline.

## What are the main results?

- Multi-turn prior context can cause dramatic degradation in MCQ accuracy:
  - up to ~73% drop for some models under certain context conditions.
  - even strong models (e.g., GPT-4o) see up to ~32% accuracy decrease.
- “Bigger is not always better”: relative ranking of large vs smaller models can change under different context regimes.
- A simple mitigation—strategic placement of the task description within the context—can improve accuracy substantially (claimed up to ~3.5×).

## How is this similar to GALILEO?

- Same general theme: evaluating *robustness over multi-turn interaction histories* rather than single-turn snapshots.
- Emphasizes protocol design (controlled context manipulations) and reporting degradation patterns, which is directly relevant for framing GALILEO’s evaluation methodology.

## How is this different from GALILEO?

- Focuses on MCQ accuracy degradation under context length/domain-shift, rather than (e.g.) agreement/stance drift, persuasion/sycophancy, or other interaction-specific failure modes.
- Appears to use a relatively “static” target task (GPQA MCQ) appended after prior turns, rather than adaptive adversarial multi-turn steering strategies.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO includes richer, interaction-native outcomes (drift, compliance, recovery, safety-relevant behaviors) and clearer causal factors, it can argue broader ecological validity beyond MCQ accuracy.
- If GALILEO reports time-to-failure / turn-of-failure style metrics, it can better capture *when* degradation occurs, not just *how much*.

## Where GALILEO is weaker / needs to improve

- GALILEO should be explicit about context controls (length, topicality/domain shift, instruction placement) to avoid confounding multi-turn effects with prompt formatting artifacts.
- Consider including an “instruction placement” ablation (early vs late vs repeated) as a simple, reproducible mitigation baseline.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a context-sensitivity ablation: vary prior-turn length and topical/domain similarity, report degradation curves.
- [ ] Add an instruction-placement baseline (and/or repeated task reminder) to test whether failures are largely due to instruction dilution.
- [ ] In related work, cite this as evidence that single-turn benchmarks are not reliable proxies for multi-turn performance.

## Quotes / details to potentially cite

- “performance on multiple-choice questions can degrade dramatically in multi-turn interactions” (abstract)
- “performance drops as large as 73% for certain models” (abstract)
- “Even highly capable models such as GPT-4o exhibit up to a 32% decrease in accuracy” (abstract)
- “strategic placement of the task description within the context can substantially mitigate performance drops … by as much as a factor of 3.5” (abstract)
