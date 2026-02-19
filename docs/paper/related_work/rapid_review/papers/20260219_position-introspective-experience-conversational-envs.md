# Position: Introspective Experience from Conversational Environments as a Path to Better Learning

- Year: 2026
- Venue: ICML (Position paper)
- Authors: Jackson Tolins; Diego Antognini; Jingling Li; Martin Klissarov; Tom Duerig
- URL: https://arxiv.org/abs/2602.14910
- BibTeX key (if we add it): tolins2026position-introspective
- Tags: introspection, conversational-environments, sensemaking, sycophancy, data-quality

## One-sentence takeaway

A position paper arguing that robust, compute-efficient “reasoning” should be trained as internalized social self-reflection learned from high-quality multi-agent dialogue, and that low-quality/ sycophantic conversational environments yield brittle internal critics.

## What problem does it solve?

- Claims that “reasoning emerges from scale” is an incomplete story; proposes that scalable improvements in robust reasoning require *training* the model’s introspective/self-reflective dialogue (internal critic/planner) via external conversational interaction.
- Motivates why simply attaching LLMs/VLMs to RL loops is insufficient: agents need a mechanism to convert sparse observations into dense, learnable “experiences” through interpretation/narrative.

## What is the core method / protocol?

- Not a concrete algorithmic contribution; it advances three core positions (framed via Vygotsky-style developmental psychology):
  - **Social genesis of the private mind**: internal reasoning as internalized “public debate” dynamics (polyphonic self: critic/planner/speaker).
  - **Introspective experience / sense-making wedge**: convert raw observations into synthetic narrative experiences and learn from the *interpretation* rather than the raw stream.
  - **Dialogue quality is the new data quality**: the rigor/diversity of dialogue mastered bounds the quality of private reasoning; sycophantic environments lead to hallucinating/agreeable internal critics.
- Connects to prior “self-reflection” work (e.g., Reflexion) but argues introspection should be *socially derived* rather than only prompt-engineered debugging.

## What are the key metrics?

- None (position paper; no reported benchmark/metrics).

## What are the main results?

- Conceptual: proposes a training paradigm shift toward optimizing conversational scaffolds/environments and dialogue quality to improve reasoning robustness and test-time compute efficiency.
- Concrete claim worth citing: *sycophantic external environments → hallucinating internal critic; adversarial/rigorous environments → robust internal reasoner*.

## How is this similar to GALILEO?

- If GALILEO studies (or mitigates) **sycophancy/agreeableness/compliance** in multi-turn settings, this paper provides a conceptual lens: sycophancy is not just an output behavior but can reflect (and shape) the quality of an internal critic learned from the agent’s interaction environment.
- Emphasizes **multi-turn conversational environments** as the substrate where robustness and “reasoning” behaviors are formed—aligned with evaluation/training setups that stress sequential interaction.

## How is this different from GALILEO?

- This work is primarily **theoretical/agenda-setting**; it does not introduce a measurable benchmark, dataset, or intervention with quantitative results.
- Focuses on broad “reasoning via introspection” rather than isolating/operationalizing sycophancy as a phenomenon with metrics and controlled experimental designs.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO is empirical (new benchmark, metrics, controlled multi-turn protocols, or mitigations), it offers the measurable evidence and operational definitions that this position paper intentionally lacks.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently treats sycophancy/compliance largely as a model-side behavior, it may under-specify the role of the **interaction environment** (e.g., “dialogue quality”, adversarial vs. agreeable partners, feedback dynamics) as a causal driver.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a short “conceptual motivation” paragraph: tie sycophancy/robustness to *training/eval environments* (dialogue quality), citing this as a position/agenda piece.
- [ ] If relevant, add an ablation dimension: vary partner behavior (sycophantic vs. critical vs. adversarial) and measure downstream robustness/consistency.
- [ ] Consider terminology: “dialogue quality” framing can contextualize why certain multi-turn settings systematically amplify sycophancy or collapse internal critique.

## Quotes / details to potentially cite

- Abstract-level thesis: robust reasoning emerges from “linguistic self-reflection” internalized from high-quality social interaction.
- Introduction framing (paraphrase): “Dialogue quality bounds reasoning quality; a sycophantic external environment leads to a hallucinating internal critic; a rigorous adversarial environment creates a robust internal reasoner.”
