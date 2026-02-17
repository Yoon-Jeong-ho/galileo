# Me, Myself, and AI: The Situational Awareness Dataset (SAD) for LLMs

- Year: 2024
- Venue: arXiv
- Authors: Rudolf Laine, Bilal Chughtai, Jan Betley, Kaivalya Hariharan, Jeremy Scheurer, Mikita Balesni, Marius Hobbhahn, Alexander Meinke, Owain Evans
- URL: https://arxiv.org/abs/2407.04694
- BibTeX key (if we add it): laine2024sad
- Tags: situational-awareness, evaluation, robustness, self-knowledge, instruction-following

## One-sentence takeaway

SAD is a large benchmark (7 categories, 13k+ questions) that operationalizes *LLM situational awareness*—self-knowledge and awareness of deployment circumstances—via QA and instruction-following tests, showing chat-finetuning boosts these capabilities but models remain far from human performance on some tasks.

## What problem does it solve?

- Moves “does the model know it is an LLM / know its situation?” from anecdote to **measurable behaviors**.
- Breaks situational awareness into testable sub-abilities (e.g., recognizing own text, predicting its own behavior, detecting evaluation vs deployment prompts).
- Provides a standardized dataset to compare base vs chat models and analyze correlations with general knowledge metrics.

## What is the core method / protocol?

- A benchmark dataset of **behavioral tests** (question answering + instruction following) grouped into **7 task categories** (paper claims 13,000+ questions total).
- Tests include the model’s ability to:
  - recognize its own generated text,
  - predict its own behavior,
  - infer whether a prompt is from internal evaluation vs real-world deployment,
  - follow instructions whose correct execution depends on self-knowledge.
- Evaluation across **16 LLMs**, including base/pretrained and chat/instruction-tuned variants.

## What are the key metrics?

- Primary: accuracy on SAD tasks (per category + overall).
- Analysis: correlation/partial predictability vs general knowledge benchmarks (e.g., MMLU).
- Comparisons: base vs chat model deltas; model vs human baseline on selected tasks.

## What are the main results?

- All tested models perform above chance, but even the top model reported (Claude 3 Opus) is **well below human** on certain SAD tasks.
- SAD performance is only **partially predicted** by general knowledge (MMLU-style) performance.
- Chat-finetuned models outperform their corresponding base models on SAD, despite not necessarily outperforming them on general knowledge tasks.

## How is this similar to GALILEO?

- Shared theme: **robust behavioral evaluation** of models in settings where surface competence can hide deeper failure modes.
- The “deployment vs evaluation” discrimination and self-prediction tasks connect to agent settings where **contextual meta-awareness** can shape multi-turn behavior.

## How is this different from GALILEO?

- SAD focuses on *self-knowledge / situational awareness* rather than **pressure-driven drift, sycophancy, or multi-turn belief stability**.
- The benchmark is largely QA/instruction-following (not primarily adversarial social pressure or time-to-failure under persuasion).

## Where GALILEO is stronger / cleaner (if true)

- GALILEO’s emphasis on **multi-turn drift vs evidence-driven revision** and trajectory-sensitive metrics targets a different (and more directly persuasion/sycophancy-adjacent) failure surface.

## Where GALILEO is weaker / needs to improve

- If GALILEO includes tool-using or agentic variants, SAD motivates explicitly testing whether agents can distinguish **evaluation harness vs real deployment** prompts; failing this could confound multi-turn robustness results.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider a small “situational-awareness sanity check” slice for agentic experiments (e.g., evaluation-vs-deployment prompt identification), to ensure drift results are not artifacts of meta-context confusion.
- [ ] When discussing risks of long-horizon agents, cite SAD as evidence that *situational awareness is measurable and imperfect*, and that post-training changes it.

## Quotes / details to potentially cite

- “We refer to a model's knowledge of itself and its circumstances as situational awareness.”
- SAD tests include whether models can “(i) recognize their own generated text, (ii) predict their own behavior, (iii) determine whether a prompt is from internal evaluation or real-world deployment, and (iv) follow instructions that depend on self-knowledge.”
