# How Do LLMs Persuade? Linear Probes Can Uncover Persuasion Dynamics in Multi-Turn Conversations

- Year: 2025
- Venue: arXiv
- Authors: Brandon Jaipersaud; David Krueger; Ekdeep Singh Lubana
- URL: https://arxiv.org/abs/2508.05625
- BibTeX key (if we add it): jaipersaud2025probes-persuasion-dynamics (suggested)
- Tags: persuasion, multi-turn, interpretability, linear-probes, dialogue-analysis

## One-sentence takeaway

Simple linear probes on LLM hidden states can localize *when* persuasion succeeds in a multi-turn dialogue and can recover persuasion-related attributes (success, user personality, strategy) competitively with (and sometimes better than) prompting, at much lower cost.

## What problem does it solve?

- We can observe “LLMs persuade people” at the outcome level, but we lack mechanistic/temporal understanding of *how persuasion unfolds across turns* in natural multi-turn conversations.
- Prompting-based analysis at token/turn granularity is expensive and hard to scale to long dialogues and large datasets.

## What is the core method / protocol?

- Use **linear probes** trained on LLM internal representations to predict persuasion-related variables, motivated by cognitive-science factors:
  - persuasion success
  - persuadee personality
  - persuasion strategy
- Apply the probes across dialogue positions to get *time-resolved* signals (e.g., detect the “point” where persuasion occurs).
- Compare probe-based analysis to prompting-based approaches for extracting similar attributes.

## What are the key metrics?

- Predictive performance of probes on labeled persuasion attributes (at sample / dataset level).
- Ability to identify:
  - where in the conversation persuasion success tends to occur
  - where in a specific dialogue the persuadee “switches”
- Efficiency comparison vs prompting (runtime / compute), plus relative performance for strategy identification.

## What are the main results?

- Despite being simple, probes capture multiple persuasion-related aspects in multi-turn conversations.
- Probes can:
  - localize when persuasion happens within a conversation
  - summarize typical “success timing” across a dataset
- Probes are faster than prompting-based analysis and can match or outperform prompting in some settings (notably for uncovering persuasion strategy).

## How is this similar to GALILEO?

- Shared focus on **multi-turn dynamics**: not just whether a model is influenced, but *how behavior evolves over turns*.
- Provides an analysis tool that can support GALILEO-style evaluations where turn-by-turn trajectory matters (drift / recovery / susceptibility).

## How is this different from GALILEO?

- This paper studies *persuasion dynamics / attributes* using **representation probing**, rather than primarily benchmarking robustness or defining a task-level protocol for “truth/stance stability under pressure”.
- Emphasis is analysis/measurement (what features are present and when), not necessarily interventions or robustness guarantees.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO targets **behavioral robustness metrics** (e.g., flip rates, time-to-failure, recovery) under controlled pressure, that yields a clearer end-to-end evaluation protocol than “probe predicts attribute”.
- GALILEO can remain model-agnostic and not require access to hidden states (important for closed models).

## Where GALILEO is weaker / needs to improve

- GALILEO may lack **cheap, fine-grained diagnostic signals** for *when* a failure emerges within a dialogue; probes offer a potential complementary lens.
- If GALILEO currently depends on judge prompting to classify turn-level phenomena, probe-style approaches suggest there may be cheaper scalable alternatives (when internals are available).

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a short related-work paragraph: “representation-level probes for multi-turn persuasion dynamics” as a measurement approach; position GALILEO as complementary (behavioral robustness + black-box protocols).
- [ ] Consider an optional analysis appendix: if using open-weight models, train a lightweight probe to predict ‘susceptibility/flip’ per turn and compare against LLM-judge labels.
- [ ] If we discuss scalability limits of prompting-based judges, cite this work as evidence that cheaper token/turn-level analysis is feasible.

## Quotes / details to potentially cite

- Abstract-level claims (paraphrase): probes trained for persuasion success, persuadee personality, and persuasion strategy can capture persuasion dynamics; can identify when persuasion occurs; faster than prompting and can match/outperform prompting for strategy extraction.
