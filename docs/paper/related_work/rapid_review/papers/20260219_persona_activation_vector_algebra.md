# PERSONA: Dynamic and Compositional Inference-Time Personality Control via Activation Vector Algebra

- Year: 2026
- Venue: ICLR 2026 (submitted/accepted per arXiv comments)
- Authors: Xiachong Feng et al. (see arXiv)
- URL: https://arxiv.org/abs/2602.15669
- BibTeX key (if we add it): persona-feng-2026
- Tags: persona, personality-control, activation-steering, representation-vectors, inference-time-control

## One-sentence takeaway

PERSONA claims that personality traits in LLMs correspond to approximately orthogonal activation-space directions that can be extracted and then composed/algebraically manipulated at inference time to control persona without fine-tuning.

## What problem does it solve?

- Prompting-based persona control is often **static** and brittle, and fine-tuning-based control is expensive.
- Existing approaches struggle with **dynamic** (context-dependent) and **compositional** (multiple traits at once) personality control.

## What is the core method / protocol?

Three-stage, training-free framework (per abstract):

- **Persona-Base:** extract trait vectors using *contrastive activation analysis*; enforce/encourage vectors to be **orthogonal** (traits as separable directions).
- **Persona-Algebra:** control by activation-vector arithmetic:
  - scalar multiplication → intensity
  - addition → composition
  - subtraction → suppression
- **Persona-Flow:** during inference, **dynamically** compose trait vectors based on context to adapt personality over time.

(Details likely resemble activation steering / residual stream editing; rapid read based on abstract only.)

## What are the key metrics?

- **PersonalityBench**: reports mean score (higher is better; scale not described in abstract).
- **Persona-Evolve** (new benchmark): pairwise win-rate for **dynamic personality adaptation**.

## What are the main results?

- On PersonalityBench: mean score **9.60**, nearly matching supervised fine-tuning upper bound **9.61**, with **no gradient updates**.
- On Persona-Evolve: up to **91% win rates** across “diverse model families”.

## How is this similar to GALILEO?

- Shares the general theme of **controlling / stabilizing behavioral traits** in LLMs, and treating behavior as something you can operationalize and measure.
- Mechanistically adjacent if GALILEO discusses *representational directions* (e.g., linear features/axes) behind behavioral drift or style/persona shifts.

## How is this different from GALILEO?

- PERSONA is primarily an **intervention** paper (activation-space steering for persona), while GALILEO (as positioned in recent rapid-review shortlist) appears more focused on **multi-turn robustness / social pressure / drift-vs-revision evaluation**.
- Requires some form of **internal activation access** (or an equivalent hook) to manipulate vectors, which may not align with GALILEO’s preferred threat models if they assume black-box access.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO is black-box / protocol-driven, it may provide a **cleaner, more deployable evaluation** that does not depend on model internals.
- If GALILEO separates *evidence-driven revision* from *pressure-driven drift*, that diagnostic framing is orthogonal to PERSONA’s control objective.

## Where GALILEO is weaker / needs to improve

- If GALILEO discusses persona/behavior drift but lacks an intervention, PERSONA is a strong example of an **efficient, training-free control knob** worth citing as “what’s possible with internal access”.

## Action items for GALILEO (experiments / method / writing)

- [ ] Related work: cite as **training-free activation-space persona control**; connect to “axes/directions” framing (also aligns with other mechanistic persona papers).
- [ ] If GALILEO includes any mechanistic section: mention the hypothesis that traits can be **approximately orthogonal directions**, enabling composition/suppression.
- [ ] If relevant: consider an ablation discussion contrasting (i) prompt-only persona control vs (ii) internal activation steering (as an upper-bound / alternative threat model).

## Quotes / details to potentially cite

- “training-free framework … direct manipulation of personality vectors in activation space”
- “personality traits appear as extractable, approximately orthogonal directions … support algebraic operations”
- “mean score of 9.60 … nearly matching … 9.61 without any gradient updates”
- “Persona-Evolve benchmark … up to 91% win rates across diverse model families”
