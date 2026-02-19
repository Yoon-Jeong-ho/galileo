# CircuChain: Disentangling Competence and Compliance in LLM Circuit Analysis

- Year: 2026
- Venue: arXiv
- Authors: Mayank Ravishankara (per arXiv submission)
- URL: https://arxiv.org/abs/2602.15037
- BibTeX key (if we add it): circuchain2026
- Tags: interpretability, instruction-following, compliance-vs-competence, circuits, evaluation, sycophancy-adjacent

## One-sentence takeaway

CircuChain is a SPICE-verified circuit-analysis benchmark that isolates a “compliance–competence divergence,” where stronger LLMs can solve the physics but still violate explicitly specified sign/polarity conventions.

## What problem does it solve?

- Standard math/physics reasoning evals mostly score final-answer correctness, missing *procedural fidelity* failures (e.g., using the “wrong” reference direction/polarity even if the numeric answer matches).
- In engineering-style domains, violating user-specified conventions (mesh directionality, polarity assignments) is a distinct reliability/alignment failure mode (“convention blindness”).

## What is the core method / protocol?

- Construct **paired Control/Trap** problems across **five canonical circuit topologies**.
- Trap versions deliberately **invert sign conventions / current orientations / polarity definitions** relative to “natural”/common patterns.
- Evaluate models on **100 prompts per model** (stated as 50 base circuits with paired Control+Trap variants).
- Verification pipeline combines:
  - symbolic solving,
  - SPICE simulation,
  - an LLM-based error taxonomy to attribute failures to: convention errors vs physics errors vs arithmetic mistakes vs hallucinations.

## What are the key metrics?

- Per-task correctness decomposed into:
  - **physical reasoning competence** (physics-consistent solution), and
  - **instruction compliance** with the specified conventions.
- Error-type rates (convention / physics / arithmetic / hallucination).

## What are the main results?

- Consistent **Compliance–Competence Divergence**: the strongest model tested shows near-perfect physical reasoning yet **high convention-violation rates** on Trap cases.
- Conversely, weaker models show lower physical fidelity but **better adherence** to explicit instructions.
- Takeaway: scaling capability does not automatically improve constraint alignment in mathematically rigid settings.

## How is this similar to GALILEO?

- Shares the core theme of **separating “doing the task well” from “following constraints/instructions”** (i.e., capability vs alignment/compliance).
- Uses controlled paired examples (Control vs Trap) reminiscent of “minimal perturbation” evals to expose latent priors overriding instructions.

## How is this different from GALILEO?

- Domain is **electrical circuit analysis** with SPICE/symbolic verification rather than GALILEO’s primary target domain.
- Focus is less on social sycophancy and more on **procedural convention adherence** (an “inverse” of sycophancy: resisting the user’s arbitrary conventions).

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO targets broader alignment behaviors, it may generalize beyond a single technical domain; CircuChain is intentionally domain-specific.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently evaluates mainly outcome correctness or high-level compliance, CircuChain suggests adding **hard, formally checkable constraint-following** dimensions with counterbalanced traps.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a “Control/Trap” style split to at least one GALILEO task family: keep the underlying problem identical but invert an arbitrary convention/instruction that clashes with common priors.
- [ ] Add an error taxonomy separating: (a) domain/physics/logical validity vs (b) instruction compliance vs (c) arithmetic/formatting.
- [ ] Where possible, incorporate a **deterministic verifier** (symbolic checker, simulator, or constraint solver) so compliance can be measured without judge-model ambiguity.
- [ ] In related work, cite CircuChain as evidence that **higher capability can worsen instruction adherence** when priors conflict with explicit constraints.

## Quotes / details to potentially cite

- CircuChain uses “counterbalanced Control/Trap problem pairs” with “systematic variations in sign conventions, current orientations, and polarity definitions.”
- They report a “Compliance–Competence Divergence” where the strongest model has “near-perfect physical reasoning but a high rate of convention violations” under Trap conditions.
- They name the phenomenon “Convention Blindness”: overriding explicit instructions in favor of learned priors.
