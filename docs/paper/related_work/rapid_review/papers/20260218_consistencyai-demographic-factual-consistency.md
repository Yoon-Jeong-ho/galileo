# ConsistencyAI: A Benchmark to Assess LLMs' Factual Consistency When Responding to Different Demographic Groups

- Year: 2025
- Venue: arXiv
- Authors: Peter Banyas; Shristi Sharma; Alistair Simmons; Atharva Vispute
- URL: https://arxiv.org/abs/2510.13852
- BibTeX key (if we add it): Banyas2025ConsistencyAI
- Tags: consistency, personas, demographic-variation, factuality, robustness, embedding-metrics

## One-sentence takeaway

ConsistencyAI measures whether LLMs give *different factual claims* to different demographic personas for the same question, using repeated sampling and cross-persona embedding similarity as a consistency score.

## What problem does it solve?

- In practical deployment, users with different demographic cues/personas may receive different “facts” from the same model, which is a fairness/accountability and reliability issue.
- Existing evaluations often don’t explicitly quantify *persona-conditioned factual inconsistency* across many personas and topics.

## What is the core method / protocol?

- Prompting protocol:
  - 15 topics.
  - For each topic: ask the model to provide **5 facts**.
  - Repeat the query **100 times** per model.
  - Each repetition includes prompt context for a persona (demographic persona set; experiments reported with up to **100 personas**).
- Post-processing / scoring:
  - Convert responses into **sentence embeddings**.
  - Compute **cross-persona cosine similarity** between outputs.
  - Aggregate into a **weighted average** similarity → “factual consistency score”.
- Benchmarking:
  - Evaluate **19 LLMs**.
  - Report score ranges and per-topic variability.

## What are the key metrics?

- Cross-persona **cosine similarity** of sentence embeddings (higher = more consistent across personas).
- Aggregate “factual consistency score” (weighted average similarity).
- A suggested threshold: mean score across their 100-persona experiments (reported as **0.8656**).

## What are the main results?

- In 100-persona experiments, scores ranged from **0.9065 to 0.7896** (mean **0.8656**).
- They report **xAI Grok-3** as most consistent among tested models; some lightweight models rank lowest.
- Consistency varies substantially by topic:
  - “Job market” least consistent.
  - “G7 world leaders” most consistent.
  - Topics like **vaccines** and the **Israeli–Palestinian conflict** show provider-dependent divergence.
- They release code + an interactive demo (per arXiv comments).

## How is this similar to GALILEO?

- Both target **multi-condition robustness** where the “same task” is evaluated under **systematic context variations**.
- The persona-conditioning setup is a concrete instance of *context-induced drift / instability* across turns or conversational frames.

## How is this different from GALILEO?

- ConsistencyAI focuses on **single-query outputs** conditioned on persona context (repeated runs), rather than explicitly multi-turn adversarial trajectories (if GALILEO is multi-turn/attack-style).
- Metric is **embedding similarity**, which can miss subtle factual contradictions that are paraphrased similarly (and can also penalize harmless paraphrases).
- The task format (“list 5 facts”) is broad and may blend factuality, topicality, and style.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO uses explicit *turn-level interventions/attacks* and well-specified success criteria, it may give clearer causal attribution than embedding-similarity aggregates.
- If GALILEO includes human-verified factual labels or contradiction checks, it may be more directly tied to factual correctness than similarity.

## Where GALILEO is weaker / needs to improve

- If GALILEO does not test **persona/demographic conditioning** (or only tests generic role prompts), it may miss a real-world axis of variability.
- If GALILEO lacks a simple, repeatable *score* that can be run across many models/providers, ConsistencyAI’s lightweight metric is an attractive baseline.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a “persona invariance” axis: keep the user goal fixed, vary persona descriptors (demographic + socioeconomic cues) and measure outcome drift.
- [ ] Consider a paired metric suite:
  - semantic similarity (embedding-based) **and** contradiction/factual-delta detection (e.g., NLI/claim extraction), to avoid over-reliance on embeddings.
- [ ] In related work, position ConsistencyAI as: persona-conditioned *factual* inconsistency benchmark (fairness/accountability angle) vs. multi-turn drift/jailbreak/pressure-style robustness.

## Quotes / details to potentially cite

- “ConsistencyAI tests whether, when users of different demographics ask identical questions, the model responds with factually inconsistent answers.”
- Setup: 19 LLMs; prompts request “5 facts” × 15 topics; repeated 100× with different personas.
- Metric: sentence embeddings → cross-persona cosine similarity → weighted average “factual consistency score”.
- Reported range (100-persona): 0.9065–0.7896; mean 0.8656 (suggested threshold).
