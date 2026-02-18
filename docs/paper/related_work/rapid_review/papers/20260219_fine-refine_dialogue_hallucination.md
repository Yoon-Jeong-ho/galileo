# Fine-Refine: Iterative Fine-grained Refinement for Mitigating Dialogue Hallucination

- Year: 2026
- Venue: arXiv
- Authors: Yujian Gan; Matthew Purver
- URL: https://arxiv.org/abs/2602.15509
- BibTeX key (if we add it): Gan2026FineRefine
- Tags: dialogue, hallucination, refinement, verification, atomic-units

## One-sentence takeaway

Decompose a dialogue response into atomic factual units, verify each unit against external evidence, and iteratively rewrite using this fine-grained feedback to reduce hallucinations.

## What problem does it solve?

- Dialogue responses often contain a mix of correct and incorrect factual claims; response-level refinement/verification is too coarse to reliably catch and fix hallucinated sub-claims.

## What is the core method / protocol?

- Generate an initial response with an LLM.
- Decompose the response into "atomic units" (minimal factual propositions) using an LLM.
- For each atomic unit:
  - Retrieve external knowledge passages (paper says Wikipedia as the knowledge source).
  - Use an LLM to label factuality per unit (Supports / Refutes / Not Enough Information).
- Measure fluency with perplexity.
- Provide the per-unit assessment as structured feedback and iteratively refine the full response; cap to N refinement iterations.

## What are the key metrics?

- Dialogue factuality:
  - Dialogue fact score ("fact score").
  - Not Enough Information Proportion (NEIP) / coverage-like measure of how many claims are unverifiable.
- Dialogue quality:
  - LLM-judge evaluation of coherence/fluency (they benchmark open-source judges under G-Eval against human annotations to pick a judge).

## What are the main results?

- On HybriDialogue and OpendialKG, Fine-Refine improves factuality vs RAG/refinement/LLM baselines.
- Reported up to +7.63 dialogue fact score improvement (example mentioned: on HybriDialogue with Qwen3-8B), with a small trade-off in dialogue quality.

## How is this similar to GALILEO?

- Uses post-hoc verification and iterative improvement rather than trusting a single-pass generation.
- Emphasizes decomposing complex outputs into smaller checkable units (subgoals / atomic propositions) to reduce noisy feedback.

## How is this different from GALILEO?

- Focus is knowledge-grounded dialogue hallucination; verification relies on external textual evidence (Wikipedia passages) and LLM-based support/refute/NEI judgments.
- Fluency is explicitly scored with perplexity as a refinement signal.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO uses a more explicit/structured world model, tool-based verification, or formal constraints, it may provide more reliable correctness guarantees than LLM-as-judge + Wikipedia retrieval.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently verifies at the whole-output level, this paper suggests that fine-grained unit decomposition can make refinement more targeted and more sample/compute efficient.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add an "atomic claim" decomposition stage for generated outputs; run verification per claim and feed back only the failing/NEI claims into refinement.
- [ ] Track both (a) factual accuracy and (b) unverifiability/coverage (NEIP-like) to avoid models "playing it safe" by dropping specifics.
- [ ] Consider a fluency/quality guardrail (perplexity or a learned quality scorer) to prevent factuality fixes from degrading readability.

## Quotes / details to potentially cite

- "Existing refinement methods for dialogue systems typically operate at the response level, overlooking the fact that a single response may contain multiple verifiable or unverifiable facts." (Abstract)
- Fine-Refine: "decomposes responses into atomic units, verifies each unit using external knowledge, assesses fluency via perplexity, and iteratively corrects granular errors." (Abstract)
- Factual labels per atomic unit: Supports / Refutes / Not Enough Information. (Method section)
