# How Do LLMs Persuade? Linear Probes Can Uncover Persuasion Dynamics in Multi-Turn Conversations

- Year: 2025
- Venue: arXiv
- Authors: Brandon Jaipersaud; David Krueger; Ekdeep Singh Lubana
- URL: https://arxiv.org/abs/2508.05625
- BibTeX key (if we add it): jaipersaud2025persuadeprobes
- Tags: persuasion, multi-turn, probing, representations, evaluation

## One-sentence takeaway

Linear probes on frozen LLM activations can cheaply localize *when* persuasion succeeds in a multi-turn dialogue and predict persuasion-related attributes (outcome, persuadee personality, rhetorical strategy), sometimes matching or beating prompt-based classifiers.

## What problem does it solve?

- Understanding “persuasion dynamics” in multi-turn conversations is hard to analyze at token/turn granularity using prompting-based analyses (too expensive to run an extra LLM call per token/turn).
- They want an efficient analysis tool to attribute persuasion success/failure points and characterize strategies/traits over time.

## What is the core method / protocol?

- Train simple **linear probes (logistic regression)** on **frozen LLM residual-stream activations** to predict:
  - persuasion outcome (successful vs unsuccessful)
  - persuadee personality (Big-5 traits, treated as 5 separate high/low binary classifiers)
  - persuasion strategy (3-way: logical vs emotional vs credibility appeals; “Aristotle’s rhetorical triangle”)
- Apply probes at different granularities (token/turn/conversation) to analyze trajectories and identify the “point of persuasion” in a dialogue.
- Compare to prompting-based baselines for similar prediction tasks; emphasize computational efficiency in multi-turn settings.

## What are the key metrics?

- Standard classification metrics for probe tasks (not fully captured in the truncated HTML excerpt); comparisons against prompting baselines.
- Qualitative/aggregate “trajectory” analyses: where in the dialogue persuasion cues/success concentrates (middle vs end turns).

## What are the main results?

- Probes can:
  - identify dialogue regions/turns where persuasion success occurs;
  - infer rhetorical strategy over time;
  - estimate persuadee personality attributes.
- They report probes are **faster** than prompting-based approaches and can be **competitive or better** in some settings (notably for strategy identification).
- Observed pattern: persuasive cues concentrate around **middle turns** for human dialogues, but shift toward **final turns** for LLM-generated dialogues (systematic difference between natural vs synthetic).

## How is this similar to GALILEO?

- Both care about **multi-turn, longitudinal behavior** and “trajectory-level” phenomena rather than single-turn scores.
- Offers a potential **measurement layer**: low-cost signals for *when* a conversation shifts (analogous to turn-of-failure / drift localization), but for persuasion.

## How is this different from GALILEO?

- This is primarily **representation probing / analysis**, not a robustness benchmark/protocol for resisting persuasion or maintaining truthfulness under pressure.
- Focuses on predicting *attributes* (strategy/personality/outcome) rather than defining adversarial pressure tests, counterfactual controls, or success criteria tied to factual consistency.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO defines explicit **pressure interventions + controlled trajectories + robustness metrics**, it provides a more direct evaluation target than “can we predict persuasion labels?”.
- GALILEO can emphasize **behavioral guarantees** (resistance/recovery) rather than interpretability-only signals.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently lacks cheap “on-trajectory” diagnostics, probes could provide **fine-grained localization** and reduce reliance on expensive judges.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding a lightweight “trajectory diagnostics” baseline: train simple classifiers (could be probes or smaller models) to predict **turn-of-shift** (e.g., first turn where stance flips / truth-degrades) and compare to judge-based detection.
- [ ] In related work, cite as: *linear probing for token/turn-level analysis of multi-turn persuasion dynamics; efficiency advantage vs prompting.*

## Quotes / details to potentially cite

- Abstract framing: probes can “identify the point in a conversation where the persuadee was persuaded”.
- Contribution claim: probes can “match or exceed” prompting-based methods while being “significantly” more computationally efficient for multi-turn, large-scale analysis.
- Reported dataset-level insight: human dialogues show persuasion cues mid-conversation; LLM-generated dialogues shift cues to final turns.
