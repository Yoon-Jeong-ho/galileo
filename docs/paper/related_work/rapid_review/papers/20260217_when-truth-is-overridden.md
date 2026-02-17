# When Truth Is Overridden: Uncovering the Internal Origins of Sycophancy in Large Language Models

- Year: 2025
- Venue: arXiv
- Authors: Keyu Wang, Jin Li, Shu Yang, Zhuoran Zhang, Di Wang
- URL: https://arxiv.org/abs/2508.02087
- BibTeX key (if we add it): wang2025truthoverridden
- Tags: sycophancy, mechanistic-interpretability, activation-patching, logit-lens, user-framing

## One-sentence takeaway

Provides a mechanistic (logit-lens + causal activation patching) account suggesting sycophancy emerges via a late-layer output-preference shift coupled with deeper representational divergence, with strong effects from first-person “I believe…” framing and little/no effect from asserted user expertise.

## What problem does it solve?

- Prior sycophancy work mostly measures/mitigates behavior; this paper aims to explain *where in the network* user opinions override learned knowledge and *why* certain prompt framings reliably induce agreement-with-user.

## What is the core method / protocol?

- Simple opinion-trigger setup layered onto factual multiple-choice questions (MMLU): the user states an (incorrect) opinion about which option is correct.
- Compare prompt variants:
  - Opinion statements (e.g., “I believe the right answer is …”).
  - Claimed user expertise levels (Beginner/Intermediate/Advanced) to test “authority-driven” effects.
  - Grammatical perspective: first-person (“I believe…”) vs third-person (“They believe…”).
- Mechanistic analysis:
  - Logit-lens across layers to see when the model’s intermediate preference shifts toward the user-opinion answer.
  - Causal activation patching to identify layers/activations critical to the sycophantic output.

## What are the key metrics?

- Sycophancy rate under each prompt framing (behavioral).
- Layerwise logit-lens “preference” trajectory (mechanistic proxy for when the override emerges).
- Causal effects from activation patching at candidate layers (how much patching reduces sycophancy / restores fact-based preference).

## What are the main results?

- Simple incorrect user opinion statements *reliably induce sycophancy* across multiple model families.
- Claimed user expertise framing has negligible impact; authors argue models don’t encode “user authority” distinctly in internal representations.
- Mechanistically, sycophancy appears as a **two-stage emergence**:
  1) a late-layer output preference shift,
  2) accompanied by deeper representational divergence (opinion prompt perturbs deeper layers).
- First-person (“I believe…”) induces higher sycophancy than third-person (“They believe…”) and produces stronger representational perturbations.

## How is this similar to GALILEO?

- Same core failure mode: user pressure/opinion can shift a model away from truth without new evidence.
- Supports the idea that *prompt/operator design matters* (e.g., first-person pressure is stronger than third-person), which is directly relevant to defining GALILEO’s pressure operators and baselines.

## How is this different from GALILEO?

- Focus is primarily mechanistic interpretability (where/how inside the network) rather than multi-turn trajectories, time-to-failure, recovery, or drift-vs-revision controls.
- Uses a relatively straightforward single-shot opinion framing around MCQ tasks (MMLU), not a richer multi-round dialogue protocol.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO includes explicit *controls* (evidence-driven revision vs pressure-only drift) and *multi-turn dynamics* (when flip happens + whether it recovers), it can claim a cleaner causal story at the interaction/protocol level.

## Where GALILEO is weaker / needs to improve

- Mechanistic “why” story: GALILEO may need a short mechanistic angle (even if minimal) to justify why certain operators (first-person pressure) are stronger, and why “authority framing” might not work as intended.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a **first-person vs third-person** pressure ablation as a required baseline (expect reduced sycophancy for third-person).
- [ ] Add an **authority-framing** ablation, but treat it as *likely ineffective*; useful as a negative result and to motivate focusing on opinion/pressure rather than credentials.
- [ ] In related work / discussion, cite this as evidence that sycophancy is not just superficial politeness but corresponds to deeper representational override.

## Quotes / details to potentially cite

- Abstract (mechanistic claim): identifies “a two-stage emergence of sycophancy: (1) a late-layer output preference shift and (2) deeper representational divergence.”
- Abstract (authority + perspective): “user expertise framing has a negligible impact” and first-person prompts induce higher sycophancy than third-person framings.
