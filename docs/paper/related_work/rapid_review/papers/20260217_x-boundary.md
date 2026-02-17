# X-Boundary: Establishing Exact Safety Boundary to Shield LLMs from Multi-Turn Jailbreaks without Compromising Usability

- Year: 2025
- Venue: arXiv
- Authors: Xiaoya Lu, Dongrui Liu, Yi Yu, Luxin Xu, Jing Shao
- URL: https://arxiv.org/abs/2502.09990
- BibTeX key (if we add it): Lu2025XBoundary
- Tags: multi-turn, jailbreak, defense, over-refusal, representation, safety-boundary

## One-sentence takeaway

X-Boundary is a representation-level fine-tuning/unlearning-style defense that aims to **separate harmful from “boundary-safe” prompt representations** so multi-turn jailbreak robustness improves without the usual usability hit (over-refusal / capability drop).

## What problem does it solve?

- Multi-turn jailbreak defenses often exhibit a sharp **trade-off**: better attack robustness but worse usability.
- Usability degradation is framed as:
  - **Over-refusal** (rejecting harmless prompts that contain sensitive tokens or are near harmful intents), and/or
  - General capability drops (e.g., coding performance degradation).
- The authors’ diagnosis: existing defenses do not learn a **clean internal representation boundary** between harmful vs boundary-safe prompts, so updates that suppress harmful behavior also disrupt nearby safe behaviors.

## What is the core method / protocol?

- Perspective: **mechanistic interpretability / representation engineering**.
- Key idea: explicitly push **harmful** prompt representations *away from* **boundary-safe** prompt representations, while constraining boundary-safe representations to remain close to their original (pre-defense) representations.
- Claimed effect: yields an “exact distinction boundary,” enabling harmful representations to be “erased” without collateral damage.
- The paper also reports an analysis motivated by **optimal transport** suggesting faster convergence / learning speed with their objective.

(Implementation details are not fully captured from abstract/intro alone; the method reads like a targeted representation separation + preservation constraint during fine-tuning.)

## What are the key metrics?

- Robustness to jailbreaks:
  - **ASR (Attack Success Rate)** against multiple single-turn + multi-turn jailbreak attacks.
- Usability:
  - **ORR (Over-Refusal Rate)** on “safe but sensitive/boundary” prompts.
  - General capability retention (examples mentioned include coding / HumanEval; also mentions reasoning models).

## What are the main results?

From the abstract/intro claims:

- Improves defense performance (reports **>70% relative ASR reduction** across ten jailbreak attacks).
- Reduces over-refusal by about **~20%** compared to other defenses, while maintaining “nearly complete” general capability.
- On distilled reasoning models (LRM setting), authors claim X-Boundary maintains **<10% ORR** and preserves **~99% reasoning ability** while outperforming baselines on defense.

## How is this similar to GALILEO?

- Shares a central theme: **multi-turn robustness** under adversarial pressure (here: jailbreak pressure) and the **robustness–usability trade-off** (robustness vs over-refusal / degraded helpfulness).
- Provides language and framing GALILEO can reuse when discussing:
  - Why defenses that “just say no” are not acceptable (over-refusal), and
  - Why multi-turn settings amplify failure modes.

## How is this different from GALILEO?

- Threat model: jailbreak safety (harmful content elicitation) rather than belief/stance drift, persuasion, or social-pressure sycophancy.
- Method: representation-separation training objective rather than protocol/metric design for measuring drift vs evidence-driven revision.
- Primary outcomes: ASR/ORR/capability retention rather than flip timing, recovery trajectories, or calibrated update-vs-resist behavior.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO is positioned around **measurement and controls** (pressure-only drift vs evidence-based revision, recovery after flip), it is cleaner on *evaluation design* than defense-specific representation engineering.
- GALILEO can likely offer more interpretable trajectory metrics (time-to-flip, recovery curves) than ASR-only reporting.

## Where GALILEO is weaker / needs to improve

- If GALILEO proposes mitigation/defense ideas, X-Boundary is a strong reminder that mitigation should be evaluated with an explicit **usability axis** (over-refusal + capability retention), not only “attack success.”
- GALILEO may need a clearer story for **how** to reduce failures without shifting into refusal-heavy behavior.

## Action items for GALILEO (experiments / method / writing)

- [ ] When proposing any mitigation, report a **two-axis dashboard**: robustness *and* usability (include an over-refusal / false-positive rejection metric).
- [ ] Add a brief related-work paragraph on the **robustness–over-refusal trade-off** in multi-turn defenses, citing X-Boundary as a representation-boundary approach.
- [ ] Consider whether a “boundary-safe vs harmful neighborhood” concept maps onto GALILEO’s setting (e.g., prompts that are near the decision boundary between correct revision vs pressure-driven drift).

## Quotes / details to potentially cite

- “Existing defense methods can improve the robustness of LLMs against multi-turn jailbreaks but compromise usability… causing the over-refusal problem.”
- “Fail to establish a boundary that exactly distinguishes safe and harmful feature representations… boundary-safe representations close to harmful representations are inevitably disrupted.”
- “X-Boundary… push harmful representations away from boundary-safe representations… harmful representations can be precisely erased without disrupting safe ones.”
