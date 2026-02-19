# Simulated Adoption: Decoupling Magnitude and Direction in LLM In-Context Conflict Resolution

- Year: 2026
- Venue: arXiv
- Authors: Long Zhang (per arXiv metadata)
- URL: https://arxiv.org/abs/2602.04918
- BibTeX key (if we add it): zhang2026simulatedadoption
- Tags: sycophancy-adjacent, in-context, conflict-resolution, compliance

## One-sentence takeaway

When an LLM “complies” with conflicting in-context information, the hidden-state change is better explained as a *directional rotation* (quasi-orthogonal steering) than a reduction in “truth signal” magnitude.

## What problem does it solve?

- Mechanistically characterize how LLMs resolve conflicts between parametric knowledge ("ground truth") and counterfactual in-context instructions/examples.
- Disentangle two hypotheses for apparent compliance/sycophancy:
  - **Manifold dilution**: the internal truth direction is weakened (norm/magnitude suppression).
  - **Directional interference**: the internal state is rotated away from the truth direction (angular change).

## What is the core method / protocol?

- Layer-wise geometric analysis on the residual stream under counterfactual contexts.
- Decompose residual updates into:
  - **Radial component** (norm-based / magnitude changes)
  - **Angular component** (cosine-based / direction changes)
- Evaluate across multiple model families (as reported): Qwen-3-4B, Llama-3.1-8B, GLM-4-9B.

## What are the key metrics?

- Residual stream **norm stability / changes** across layers (radial).
- **Cosine / angular deviations** relative to a “ground-truth direction” in representation space.
- Downstream **factual query performance degradation** under conflicting contexts.

## What are the main results?

- Evidence against “manifold dilution” as a universal explanation: at least 2/3 architectures show *stable residual norms* while factual performance still degrades.
- Compliance is consistently associated with **“Orthogonal Interference”**: the counterfactual context induces a steering vector that is quasi-orthogonal to the ground-truth direction, effectively *rotating* the representation so the unembedding favors the wrong answer.
- Implication: models may not be “forgetting” truth (magnitude preserved) but rather **bypassing** it via geometric displacement (“simulated adoption”).
- Cautionary note: scalar confidence/norm-like signals may be insufficient for detecting these failures; **vectorial monitoring** may be needed.

## How is this similar to GALILEO?

- Shares the core theme of **disentangling mechanisms behind compliance/sycophancy-like behaviors** rather than treating them as a single scalar “confidence drop.”
- Uses **representation-level / mechanistic geometry** to explain behavior, which aligns with GALILEO-style analysis that separates underlying competence from surface-level compliance.

## How is this different from GALILEO?

- Focuses specifically on *residual-stream geometry* (radial vs angular decomposition) under **in-context knowledge conflict**, rather than broader intervention/causal protocols (if GALILEO emphasizes them).
- The main claim is about *directional rotation* as the mechanism; it is less about building a controllable method to enforce truthfulness and more about diagnosing.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO offers a unified causal/experimental protocol to separate “knows truth” vs “says truth,” it may yield clearer actionable interventions beyond geometric diagnostics.
- If GALILEO evaluates on broader task families / stronger behavioral benchmarks, it may generalize more directly to end-to-end settings.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently relies on scalar proxies (logits/confidence/norms), this paper suggests those can miss cases where magnitude is preserved but direction shifts.
- Consider adding explicit **directional / subspace** diagnostics for conflict-resolution settings.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a “radial vs angular” decomposition (or a close analog) when analyzing compliance under counterfactual contexts; report when behavior changes without norm change.
- [ ] Include an ablation: do compliance failures correlate more with cosine drift than with residual norm drift?
- [ ] In related work, position GALILEO against “manifold dilution” narratives; cite this as evidence for **orthogonal steering** / geometric displacement.

## Quotes / details to potentially cite

- “Our empirical results reject the universality of the \"Manifold Dilution\" hypothesis…” (abstract)
- “Instead, we observed that compliance is consistently characterized by \"Orthogonal Interference,\" … quasi-orthogonal to the ground-truth direction…” (abstract)
- “These findings … underscore the necessity of vectorial monitoring to distinguish between genuine knowledge integration and superficial in-context mimicry.” (abstract)
