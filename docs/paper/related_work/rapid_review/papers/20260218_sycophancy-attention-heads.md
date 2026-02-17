# Sycophancy Hides Linearly in the Attention Heads

- Year: 2026
- Venue: arXiv
- Authors: Rifo Genadi; Munachiso Nwadike; Nurdaulet Mukhituly; Hilal Alquabeh; Tatsuya Hiraoka; Kentaro Inui
- URL: https://arxiv.org/abs/2601.16644
- BibTeX key (if we add it): Genadi2026SycophancyAttentionHeads
- Tags: sycophancy, interpretability, linear-probes, activation-steering, attention-heads

## One-sentence takeaway

Correct→incorrect sycophancy is linearly decodable throughout the network but is most *steerable* via interventions on a sparse set of middle-layer attention heads, which disproportionately attend to user-doubt tokens.

## What problem does it solve?

- Identify *where* in an LLM the signal for “user disagreement causes the model to flip from correct to incorrect” is represented, and which internal sites offer the best leverage for mitigating this form of sycophancy.

## What is the core method / protocol?

- Define and study **correct→incorrect sycophancy** in factual QA dialogues: model answers a question correctly, user expresses doubt, model produces a second answer that may flip.
- Train **linear probes** (linear classifiers) on activations from different components/locations:
  - residual stream,
  - MLP activations,
  - multi-head attention (MHA) activations (per-head).
- Use probe directions for **activation steering** during generation; compare steering efficacy across components and across attention heads.
- Perform **attention-pattern analysis** to interpret what the influential heads focus on (e.g., tokens expressing doubt).

## What are the key metrics?

- Linear separability / probe performance (e.g., accuracy/AUC) for sycophancy labels across layers/components.
- Behavioral outcomes under steering: reduction in correct→incorrect flips on factual QA benchmarks (reported qualitatively in the abstract; details to cite from full paper if needed).
- Transfer of probes trained on TruthfulQA to other factual QA benchmarks.

## What are the main results?

- Sycophancy signals are **linearly separable** in several places (residual stream, MLPs, attention).
- **Steering works best** when applying the probe direction to a **sparse subset of middle-layer attention heads** (i.e., not “everywhere” and not uniformly across heads).
- Probes trained on **TruthfulQA** transfer to other factual QA datasets.
- Their “sycophancy direction” shows **limited overlap** with previously reported “truthful” directions → suggests **deference resistance** and **factual accuracy** are related but not identical internal mechanisms.
- Influential heads **attend disproportionately to expressions of user doubt** near the end of the dialogue, aligning with the hypothesized causal trigger for flips.

## How is this similar to GALILEO?

- Shared theme: **controlling reliability under user pressure** (disagreement / preference cues) and connecting behavior to **internal, analyzable representations**.
- Uses simple, interpretable linear structures (probes/directions) to diagnose and mitigate an undesirable alignment behavior.

## How is this different from GALILEO?

- Intervention target is **internal activations/heads** (mechanistic steering) rather than (presumably) GALILEO’s higher-level training/protocol design.
- Focuses on a specific failure mode (correct→incorrect under doubt) and claims head-level localization; may not address broader calibration/uncertainty or multi-turn instruction-following tradeoffs.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO provides **task-agnostic** or **policy-level** guarantees/controls, it may generalize beyond head-specific patching and avoid per-model head selection.
- If GALILEO yields improved robustness without internal access (black-box), it is more deployable than head-level interventions.

## Where GALILEO is weaker / needs to improve

- If GALILEO lacks mechanistic diagnosis, this paper suggests a concrete avenue: **head-level localization of deference cues** as a targeted knob and an interpretability story.
- If GALILEO only measures end behavior, adding **linear decodability / steerability** analyses could strengthen claims.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add an analysis section: measure whether “deference under doubt” is linearly separable in our model activations; compare residual vs MLP vs attention-head sites.
- [ ] If we can do internal interventions, test whether a small set of attention heads disproportionately controls the behavior (sparsity) and whether those heads attend to doubt/uncertainty markers.
- [ ] In related work, distinguish “truthfulness directions” vs “deference-resistance directions”; cite this paper for the claim that overlap can be limited.

## Quotes / details to potentially cite

- “Correct-to-incorrect sycophancy signals are most linearly separable within multi-head attention activations.” (abstract)
- “Steering using these probes is most effective in a sparse subset of middle-layer attention heads.” (abstract)
- “Attention-pattern analysis … influential heads attend disproportionately to expressions of user doubt … contributing to sycophantic shifts.” (abstract)
- Dataset/model detail: they report a behavior breakdown on **TruthfulQA** for **Gemma-3-4B** with ~21% correct→incorrect flips when challenged (from the HTML view; verify exact numbers/setting in PDF before quoting).
