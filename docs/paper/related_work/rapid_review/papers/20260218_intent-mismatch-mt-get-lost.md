# Intent Mismatch Causes LLMs to Get Lost in Multi-Turn Conversation

- Year: 2026
- Venue: arXiv
- Authors: Geng Liu et al. (see arXiv for full list)
- URL: https://arxiv.org/abs/2602.07338
- BibTeX key (if we add it): liu2026intentmismatch (tentative)
- Tags: multi-turn, intent-inference, drift, recovery, mediator, instruction-clarification

## One-sentence takeaway

The paper argues that the “Lost in Conversation” (LiC) degradation in multi-turn prompting is driven by an **intent alignment gap** (ambiguity in conversational context) and proposes a **Mediator–Assistant** architecture that turns vague multi-turn user intent into explicit instructions, improving multi-turn performance.

## What problem does it solve?

- Explains why LLMs can underperform in **multi-turn** interactions vs a single fully-specified instruction (“Lost in Conversation”).
- Reframes the failure: not purely model unreliability/capability, but **intent mismatch / ambiguity** between user follow-ups and model interpretation.

## Core idea / method

- **Claim:** scaling model size or training alone can’t fully fix LiC because the issue is structural ambiguity in conversational context.
- **Mediator–Assistant:**
  - a **Mediator** uses experience/historical interaction patterns to rewrite/explicate the user’s intent into a clearer, well-structured instruction;
  - an **Assistant** executes the clarified instruction.

## Evidence / results (from abstract)

- Reports significant mitigation of multi-turn performance drop across diverse LLMs.

## Relevance to GALILEO

- Useful as a related-work anchor for **multi-turn degradation as intent ambiguity** rather than “reasoning failure”.
- Conceptually adjacent to any GALILEO component that does **state/intent distillation**, **instruction stabilization**, or **plan repair** over turns.

## Potential citation hooks (from abstract)

- Introduces / discusses “Lost in Conversation (LiC)”.
- “Scaling model size or improving training alone cannot resolve this gap… structural ambiguity in conversational context.”
- “Decouple intent understanding from task execution through a Mediator-Assistant architecture.”

## Action items for GALILEO

- [ ] Decide whether to cite this as evidence that **interaction protocol** matters (prompting/interface) beyond model scaling.
- [ ] If we have a component that compresses multi-turn context, compare it to their Mediator framing.
- [ ] Consider adding a short paragraph in Related Work: LiC → intent mismatch → mediator-based clarification.
