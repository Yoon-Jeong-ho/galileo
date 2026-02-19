# Surgical Activation Steering via Generative Causal Mediation

- Year: 2026
- Venue: arXiv
- Authors: Aruna Sankaranarayanan; Amir Zur; Atticus Geiger; Dylan Hadfield-Menell
- URL: https://arxiv.org/abs/2602.16080
- BibTeX key (if we add it): sankaranarayanan2026gcm
- Tags: sycophancy, activation-steering, causal-mediation, attention-heads, interpretability

## One-sentence takeaway

A causal-mediation-based head-selection method (GCM) reliably finds a sparse set of attention heads whose activation patching/steering controls *long-form* behaviors like refusal and sycophancy better than probe-based baselines.

## What problem does it solve?

- Activation steering methods often localize “concepts” using single-token (or short-span) proxies, which breaks for behaviors that are diffuse across many tokens (e.g., sycophancy expressed throughout a paragraph, or global writing style).
- Need a way to (i) *localize* internal components that causally mediate a *long-form* behavioral difference and (ii) use them for inference-time steering.

## What is the core method / protocol?

- Construct a *contrastive* dataset of tuples: (p_orig, r_orig, p_contrast, r_contrast), where the contrastive prompt induces the target concept and the original does not.
- For each candidate component (primarily attention heads across layers):
  - Run model on p_orig.
  - “Patch” that head’s activation to the value it would have under p_contrast.
  - Measure the *indirect effect* as increased relative probability of generating r_contrast vs r_orig when conditioned on p_orig.
- Rank heads by indirect effect; select top-k% heads as causal mediators.
- Steer by intervening on the selected head set (paper discusses patching / vector-addition style activation steering; also includes an attribution-patching variant to approximate interventions).

## What are the key metrics?

- Concept scoring via an auxiliary judge model (used to validate presence/absence of concept in generated long-form text).
- Steering efficacy comparisons vs baselines:
  - Random head selection.
  - Linear-probe-based head selection (correlational).
- Task-specific success criteria across:
  - Refusal induction.
  - Sycophancy reduction.
  - Verse style transfer.

## What are the main results?

- GCM identifies attention heads that act as strong *causal mediators* for long-form concepts.
- Using a sparse set of top-ranked heads, GCM steering consistently outperforms correlational probe-based baselines and random selection across multiple model families.
- Demonstrates a workable recipe for bridging: “contrastive behavioral examples” → “internal causal mediators” → “inference-time control”.

## How is this similar to GALILEO?

- Targets multi-turn / interaction-relevant failure modes like sycophancy, and treats them as behaviors that need robust measurement and mitigation.
- Uses contrastive setups (concept present vs absent) consistent with evaluation paradigms that isolate behavior changes under pressure/conditions.

## How is this different from GALILEO?

- Focuses on *mechanistic localization + intervention* (attention-head-level) rather than primarily on benchmark design / multi-turn evaluation protocols.
- Relies on an auxiliary judge model to score long-form generations, whereas GALILEO may emphasize human-grounded or task-grounded metrics (depending on the specific GALILEO setup).

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO’s contribution is chiefly evaluation/benchmarking, it can remain model-agnostic and avoid reliance on internal-access interventions.
- GALILEO can cover multi-turn dynamics end-to-end (trajectory-level failure), while GCM is primarily about steering *within a generation* conditioned on a prompt pair.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently lacks mechanistic mitigation baselines, GCM provides a concrete internal-intervention baseline for “can we *reduce* sycophancy, not just measure it?”.
- If GALILEO metrics are mostly outcome-based, it may miss opportunities to connect failures to specific internal components (useful for interpretability claims).

## Action items for GALILEO (experiments / method / writing)

- [ ] Add “GCM-style causal mediator selection” as a related-work pointer for mitigation/steering of sycophancy (esp. long-form).
- [ ] Consider a small experiment section/appendix: compare GALILEO’s preferred mitigation (if any) vs an activation-steering baseline on a shared sycophancy set.
- [ ] In writing: highlight that sycophancy is often *diffuse across turns/tokens*, and cite GCM as an approach explicitly designed for diffuse long-form concepts.

## Quotes / details to potentially cite

- Abstract (problem framing): “Where should we intervene in a language model (LM) to control behaviors that are diffused across many tokens of a long-form response?”
- Abstract (method): “We introduce Generative Causal Mediation (GCM)… to steer a binary concept … from contrastive long-form responses.”
- Abstract (tasks): “We evaluate GCM on three tasks—refusal, sycophancy, and style transfer—across three language models.”
