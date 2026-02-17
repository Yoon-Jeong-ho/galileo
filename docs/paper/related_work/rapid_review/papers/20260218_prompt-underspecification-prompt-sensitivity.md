# Revisiting Prompt Sensitivity in Large Language Models for Text Classification: The Role of Prompt Underspecification

- Year: 2026
- Venue: arXiv
- Authors: Branislav Pecher (per arXiv listing)
- URL: https://arxiv.org/abs/2602.04297
- BibTeX key (if we add it): pecher2026promptunderspecification
- Tags: prompt-sensitivity, underspecification, robustness, prompting, diagnostics

## One-sentence takeaway

A large fraction of “prompt sensitivity” reported for LLM zero/few-shot classification appears to be an artifact of underspecified prompts; adding explicit instructions and output constraints substantially reduces variance, while internal representations change only marginally (effects concentrate in late layers).

## What problem does it solve?

- Clarifies *why* small prompt changes can cause large swings in zero-/few-shot text classification performance.
- Argues that many sensitivity studies use prompts that are too vague (underspecified) and therefore overestimate true prompt brittleness.

## What is the core method / protocol?

- Construct two prompt families for classification:
  - **Underspecified prompts**: minimal task instruction, weakly constrained output space.
  - **Instruction prompts**: more explicit task instruction + tighter output constraints.
- Compare sensitivity across prompt variants using:
  - **Performance analysis** (variance across prompt variants).
  - **Logit analysis** (logit values for relevant class tokens).
  - **Linear probing** across layers to see whether underspecification meaningfully changes internal representations.

## What are the key metrics?

- Variance (or spread) of classification performance across prompt variants.
- Logit magnitude / separation for label-relevant tokens (as a proxy for how “committed” the model is to the intended output space).
- Linear probe performance / separability as a layerwise diagnostic.

## What are the main results?

- Underspecified prompts show **higher performance variance** across prompt variants.
- Underspecified prompts produce **lower logits** for relevant label tokens (suggesting weaker constraint/commitment to the intended label space).
- Linear probing suggests **only marginal differences** in internal representations; the underspecification effect emerges mainly in the **final layers**.

## How is this similar to GALILEO?

- Same general theme: separating *real* behavioral brittleness from artifacts of evaluation setup.
- Supports the idea that robustness evaluations should control for “prompt degrees of freedom” (e.g., output format, allowed label space) to avoid conflating underspecification with instability.

## How is this different from GALILEO?

- Focuses on (mostly) **single-turn** text classification prompting, not multi-turn drift/pressure dynamics.
- Studies prompt sensitivity via instruction/output-space specification rather than conversational trajectories, adversarial followups, or turn-of-failure curves.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO’s protocols already use tight output constraints and explicit instructions, they are less likely to accidentally measure “underspecification variance.”
- GALILEO’s multi-turn setups can test robustness beyond classification-style label selection.

## Where GALILEO is weaker / needs to improve

- If any GALILEO benchmarks use open-ended “answer freely” prompts but score discrete outcomes, they may be vulnerable to the same underspecification artifact.
- Might lack a simple diagnostic akin to “label-token logit margin” that flags when output-space constraints are too weak.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add an explicit “prompt specification level” ablation: underspecified vs tightly specified prompts for the same task, and report variance.
- [ ] When possible, constrain outputs (schema / fixed label set / verifier) so the evaluation measures robustness, not ambiguity.
- [ ] Track a cheap diagnostic: label-token logit margin / confidence proxy (where applicable) to detect underspecification.
- [ ] In related-work writing: cite this as evidence that some prompt sensitivity claims are evaluation artifacts, and that stronger prompt specification is a valid control condition.

## Quotes / details to potentially cite

- “a significant portion of the observed prompt sensitivity can be attributed to prompt underspecification.”
- Underspecified prompts: higher performance variance; lower logits for relevant tokens.
- Linear probing suggests marginal impact on internal representations; effects emerge in final layers.
