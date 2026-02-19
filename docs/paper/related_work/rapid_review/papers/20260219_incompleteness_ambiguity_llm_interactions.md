# An Empirical Study of the Role of Incompleteness and Ambiguity in Interactions with Large Language Models

- Year: 2025
- Venue: arXiv
- Authors: Riya Naik; Ashwin Srinivasan; Swati Agarwal; Estrid He
- URL: https://arxiv.org/abs/2503.17936
- BibTeX key (if we add it): naik2025incompleteness
- Tags: multi-turn, ambiguity, incompleteness, question-asking, robustness, interactive-qa

## One-sentence takeaway

They propose a simple neural-symbolic interaction model that operationalizes when a user query is *incomplete* or *ambiguous* from the dialogue trace, and show empirically that multi-turn interaction is especially beneficial/necessary on QA datasets with higher rates of these properties.

## What problem does it solve?

- We know multi-turn interaction can improve LLM answers, but *when* is interaction actually needed (vs single-shot answering)?
- They target question-answering settings where failures arise because the initial question is underspecified:
  - **Incompleteness**: missing required information/constraints.
  - **Ambiguity**: multiple plausible interpretations/answer conditions.

## What is the core method / protocol?

- Introduce a **message-based interaction framework** (human agent ↔ machine agent) with typed message strings (question/answer/statement/termination).
- Define properties of a question (incompleteness, ambiguity) as **deducible from the exchanged messages** over an interaction (i.e., dialogue-trace-based characterization, not just static linguistic analysis).
- Run an empirical study on benchmark QA problems, comparing outcomes as interaction length increases:
  - whether questions exhibit incompleteness/ambiguity per their criteria
  - whether answer correctness improves with additional turns

## What are the key metrics?

- Answer correctness / success on benchmark QA instances.
- Prevalence of (operationalized) **incompleteness** and/or **ambiguity** in the interaction.
- Interaction length (number of turns) vs measured incompleteness/ambiguity.

## What are the main results?

- Datasets with a high proportion of incomplete or ambiguous questions tend to **require multi-turn interaction** to reach correct answers or to establish unanswerability.
- Increasing interaction length tends to **reduce measured incompleteness/ambiguity** (presumably via clarification and constraint/specification exchange).
- Their incompleteness/ambiguity measures can be used as **characterization tools** for LLM QA interactions.

## How is this similar to GALILEO?

- Aligns with GALILEO’s interest in **robustness under underspecification** and the role of **clarification / interactive querying**.
- Suggests a framing where success depends on detecting “need-to-ask” conditions (incomplete/ambiguous user intent) rather than only better single-turn reasoning.

## How is this different from GALILEO?

- Appears more **conceptual / modeling-oriented** (a formal messaging system + properties) than building an end-to-end agentic system.
- Focuses on *classifying/characterizing* interactions and correlating with correctness, rather than proposing a concrete prompting/policy algorithm for question-asking.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO includes an explicit question-asking policy + end-to-end evaluations, it can demonstrate **actionable improvements** beyond characterization.
- GALILEO can integrate tool use / environment interaction, whereas this work seems centered on QA dialogue.

## Where GALILEO is weaker / needs to improve

- If GALILEO lacks explicit metrics for “how incomplete/ambiguous was the query?” this paper offers a direction for **measurable diagnostics** tied to interaction traces.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a short related-work paragraph: multi-turn benefit is linked to prevalence of incompleteness/ambiguity; cite as motivation for explicit clarification mechanisms.
- [ ] Consider introducing a lightweight *trace-derived* diagnostic in experiments: detect underspecification signals and correlate with when interaction helps.
- [ ] If we already do clarification, report ablations by “underspecification bucket” (high vs low ambiguity/incompleteness) to show interaction is targeted.

## Quotes / details to potentially cite

- “Our results show multi-turn interactions are usually required for datasets which have a high proportion of incompleteness or ambiguous questions; and that increasing interaction length has the effect of reducing incompleteness or ambiguity.” (Abstract)
