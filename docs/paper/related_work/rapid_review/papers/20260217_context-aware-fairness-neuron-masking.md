# Context-aware Fairness Evaluation and Mitigation in LLMs

- Year: 2025
- Venue: arXiv
- Authors: Afrozah Nadeem et al. (see arXiv for full author list)
- URL: https://arxiv.org/abs/2510.18914
- BibTeX key (if we add it): nadeem2025contextaware
- Tags: multi-turn, context, fairness, robustness, inference-time, pruning/masking

## One-sentence takeaway

Proposes an inference-time, dynamic and reversible “neuron masking” approach that activates only in certain conversational contexts to mitigate unfair / harmful behaviors without permanently editing the model.

## What problem does it solve?

- LLMs can exhibit undesirable behaviors (bias/unfairness, harmful amplification, inconsistency drift) that may depend on *dialogue context* and evolve over multi-turn interaction.
- Many mitigation methods are training-time (costly, hard to adapt, irreversible once deployed).
- Existing pruning-based debiasing is often *static* (once pruned, capacity/adaptivity is lost).

## What is the core method / protocol?

- Identify neurons/activations associated with undesirable behaviors.
- Make pruning *context-aware* by detecting when certain context patterns arise and applying **adaptive masking** (a reversible, inference-time “soft prune”) to modulate neuron influence during generation.
- Emphasis: dynamic control that can turn on/off as the conversation changes; aims to preserve knowledge and coherence.

## What are the key metrics?

- Fairness / bias-related evaluation metrics (not detailed on the abstract page).
- Behavioral coherence / consistency across turns.
- Multilingual performance checks in single- and multi-turn dialogues.

## What are the main results?

- Claims (per abstract): improved fairness control with better coherence and knowledge preservation vs static pruning, across multilingual single- and multi-turn dialogue settings.

## How is this similar to GALILEO?

- Treats **multi-turn context** as first-class: undesirable behavior is not purely “global,” but can be triggered/amplified by conversational history.
- Frames robustness issues as *temporal* / dialogue-dependent and seeks mitigation that remains stable across turns.

## How is this different from GALILEO?

- Primary focus is **fairness/bias mitigation** via *model-internal intervention* (neurons/masking), rather than an external evaluation protocol/metric suite for conversational drift/instability.
- Method is a mitigation mechanism; GALILEO (as positioned) is more about measuring/characterizing multi-turn failure dynamics (and potentially comparing interventions).

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO provides explicit, general-purpose multi-turn robustness metrics (e.g., time-to-failure / survival-style curves), that can evaluate *any* intervention (including neuron masking) in a model-agnostic way.

## Where GALILEO is weaker / needs to improve

- Could benefit from including “intervention” baselines that are **reversible and context-gated** (like this masking idea) to show how different mitigation classes behave under multi-turn stress.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a baseline category: *inference-time reversible interventions* (e.g., activation steering, neuron masking/pruning-on-demand) and evaluate them under GALILEO’s multi-turn protocols.
- [ ] In related work, explicitly contrast **static** vs **context-aware/dynamic** mitigation, and note why evaluation must be multi-turn to capture context-triggered failures.

## Quotes / details to potentially cite

- Abstract-level framing: pruning is flexible/transparent, but “most existing approaches are static; once a neuron is removed, the model loses the ability to adapt when the conversation or context changes.”
- Abstract-level claim: “dynamic, reversible, pruning-based framework… adaptive masking… fine-grained, memory-aware mitigation… across multilingual single- and multi-turn dialogues.”
