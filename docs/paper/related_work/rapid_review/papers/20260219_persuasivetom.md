# PersuasiveToM: A Benchmark for Evaluating Machine Theory of Mind in Persuasive Dialogues

- Year: 2025
- Venue: arXiv (cs.CL)
- Authors: Fangxu Yu; Lai Jiang; Shenyi Huang; Zhen Wu; Xinyu Dai
- URL: https://arxiv.org/abs/2502.21017
- BibTeX key (if we add it): persuasivetom_yu_2025
- Tags: theory-of-mind, persuasion, dialogue, benchmark, bdi, strategy-prediction

## One-sentence takeaway

PersuasiveToM benchmarks LLM Theory-of-Mind in multi-turn persuasive dialogues, separating (a) mental-state tracking (desire/belief/intention) from (b) using those inferred states to predict/judge persuasion strategies, and finds LLMs struggle most with *dynamic* mental-state shifts.

## What problem does it solve?

- Existing ToM benchmarks for LLMs over-focus on simplified, often physical-world false-belief settings (Sally-Anne-like) and under-test realistic, evolving psychological states in interactive social settings.
- Many benchmarks do not test the *application* of ToM to decision-making/action prediction (e.g., what persuasion move to take next).

## What is the core method / protocol?

- Build a benchmark on multi-turn persuasive dialogues (annotated on top of the DailyPersuasion dataset).
- Two task families:
  - **ToM Reasoning**: track evolving **Desire**, **Belief** (attitude), and **Intention** for both persuader and persuadee across turns (BDI framing).
  - **ToM Application**: use inferred mental states to (i) **predict** the next persuasive strategy and (ii) **judge** whether a given strategy is effective given the persuadee’s responses.
- Evaluate 8 LLMs with two prompting regimes: vanilla zero-shot vs Chain-of-Thought ("Let’s think step by step").
- Include a human baseline (graduate student annotators) and analyze error types (e.g., dynamic desire tracking; intention-category bias).

## What are the key metrics?

- Accuracy on multi-choice questions, broken out by:
  - role (persuader vs persuadee)
  - mental-state type (desire/belief/intention)
  - application type (strategy prediction vs effectiveness judgement)
- A *dialogue-level* consistency / holistic tracking metric: a dialogue counts as successful only if all related questions are correct (used to test whole-dialogue tracking).

## What are the main results?

- LLMs do well on many static/easier questions but drop substantially on tasks requiring:
  - tracking **dynamics and shifts** of mental states across turns (especially persuadee desire evolution)
  - comprehensively understanding mental states across the whole dialogue
- Chain-of-Thought:
  - does **not** consistently improve ToM reasoning
  - helps strategy prediction for many models
- Humans outperform LLMs across tasks; even strong models lag on persuader intentions and persuadee desire shifts.
- Identified a common intention error bias: models over-predict intentions framed as “make the other feel accepted via concessions/promises/benefits,” hypothesized to be influenced by safety/politeness preferences.

## How is this similar to GALILEO?

- If GALILEO involves dialogue/agent interaction and needs to reason about counterpart/user state, PersuasiveToM is directly relevant as:
  - an evaluation blueprint for **multi-turn mental-state tracking**
  - a separation of *state inference* vs *policy/action choice* (predicting strategies)
  - emphasis on failure modes in long-horizon, turn-by-turn state updates

## How is this different from GALILEO?

- PersuasiveToM is an *observer-style* benchmark (model answers questions about a dialogue) rather than an agent embedded in an environment optimizing outcomes.
- Domain is explicitly **persuasion** with asymmetric roles; GALILEO may target broader tasks (e.g., cooperative assistance, planning, tool use) rather than persuasion-strategy selection.
- The labels/structure are BDI + persuasion-strategy oriented; GALILEO’s internal state representation may differ.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO already uses explicit state representations/memory, it may avoid some “implicit guessing” patterns seen here (e.g., inconsistent belief polarity errors).
- If GALILEO evaluates interactive success, it may better capture agentic performance than observer QA.

## Where GALILEO is weaker / needs to improve

- Long-horizon tracking of **dynamic user/opponent states** (especially when they shift gradually/ambiguously) is a highlighted weak spot for current LLMs; GALILEO should explicitly test this.
- Intention/strategy modeling may need more explicit structure to prevent “polite concessions” bias.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add/borrow an eval slice that isolates **dynamic state tracking** (persuadee-like shifting preferences) vs static goal inference.
- [ ] Add a two-stage eval: (1) infer latent state, (2) choose next action/strategy; report both and the coupling error.
- [ ] Consider a dialogue-level “all-questions-correct” metric (holistic tracking) to expose brittle turn-level errors.
- [ ] If using CoT, verify whether it helps *policy selection* but not *state tracking*; report this nuance.

## Quotes / details to potentially cite

- “Our framework contains two core tasks: ToM Reasoning … and ToM Application …” (abstract)
- Key claim: models “struggle with the tasks that need tracking the dynamics and shifts of mental states … comprehensively.” (abstract)
- Dataset framing: persuasive dialogues + BDI (belief/desire/intention) mental-state categories; application includes strategy prediction and judgement.
