# When Meaning Stays the Same, but Models Drift: Evaluating Quality of Service under Token-Level Behavioral Instability in LLMs

- Year: 2025
- Venue: arXiv
- Authors: Xiao Li, Joel Kreuzwieser, Alan Peters
- URL: https://arxiv.org/html/2506.10095
- BibTeX key (if we add it): li2025pbss (suggested)
- Tags: drift, stability, robustness, prompt-variance, semantic-similarity, qos

## One-sentence takeaway

PBSS is a simple, black-box diagnostic that measures how much an LLM’s *outputs* drift (in embedding space) when given semantically-equivalent prompt paraphrases, and it shows strong model-specific “QoS instability under rewording” patterns.

## What problem does it solve?

- In deployed settings, users rephrase prompts; even when intent is unchanged, LLM outputs can shift in tone/structure/hedging/factual framing.
- Existing evaluations focus on correctness on a single prompt form, missing a “consistency under meaning-preserving rewordings” axis.
- The paper frames this as a quality-of-service (QoS) reliability problem: do you get the same kind of answer when you ask the same thing in different words?

## What is the core method / protocol?

- Define “prompt variance” as output changes induced by surface-level paraphrases with preserved intent.
- Construct controlled prompt-variant sets across 10 constrained tasks:
  - Start from a canonical instruction; generate multiple paraphrases.
  - Validate paraphrases with manual review + rule-based filters + semantic similarity thresholding (they mention using all-mpnet-base-v2 for this filtering).
- Query multiple LLMs under two decoding temperatures (reported: T=0.2 and T=1.3).
- Embed each output using sentence encoders (they use three SBERT-family encoders: MiniLM-L6, MiniLM-L12, all-mpnet-base-v2).
- Compute pairwise cosine distances between embeddings of outputs for all prompt-paraphrase pairs; aggregate as:
  - A drift matrix / heatmap (PBSS)
  - CDFs of drift scores
  - Visualization (t-SNE) to show clustering by task and/or model
  - Nonparametric significance tests (Kruskal–Wallis) across model groups

## What are the key metrics?

- PBSS drift score: 1 - cosine_similarity( embed(output_i), embed(output_j) ) over semantically-equivalent prompt variants.
- Distributional summaries (CDF) of drift over prompt-pairs.
- Significance testing of drift differences across model groups/encoders via Kruskal–Wallis.

## What are the main results?

- Drift under paraphrasing is systematic and model-specific; not “just noise.”
- Instruction-tuned / more modern models show tighter drift distributions (more stable responses) than older “legacy” models.
- Rankings/patterns appear relatively consistent across different embedding encoders, suggesting the signal is not an artifact of a single sentence encoder.
- They interpret the change as a qualitative “phase boundary”: older models show broad instability while newer/aligned ones cluster more tightly.

## How is this similar to GALILEO?

- Directly aligned with robustness/stability evaluation: sensitivity to superficial prompt perturbations is a key “agent QoS” dimension.
- Uses a *black-box*, post-hoc evaluation protocol that could be adapted to GALILEO-style agents: treat paraphrased user goals as perturbations and measure response drift.
- Emphasizes *behavioral consistency* and reliability rather than single-shot task accuracy.

## How is this different from GALILEO?

- PBSS evaluates single-turn LLM response drift using embedding similarity; GALILEO may care about:
  - multi-step agent trajectories
  - tool-using behavior and environment state
  - downstream task success and safety constraints
- PBSS is reference-free (no ground truth); GALILEO may need success-based metrics (task completion, constraint satisfaction) in addition to semantic similarity.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO includes environment-grounded metrics, it can detect “same semantics, different actions” failures that embedding similarity would miss.
- GALILEO can define perturbations beyond paraphrase (state perturbations, tool latency/failure, memory noise) and evaluate end-to-end robustness.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently lacks a systematic paraphrase-robustness battery, PBSS-style prompt-variant grids are a lightweight way to add it.
- Need a principled way to separate acceptable stylistic variation from harmful behavioral drift (e.g., safety tone changes, hedging, refusal behavior).

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a “semantic paraphrase perturbation suite” for key tasks: generate 10-20 paraphrases per instruction, validated by a semantic filter.
- [ ] Define multi-metric drift: (a) semantic embedding drift, (b) action/plan drift (tool calls), (c) outcome drift (task success).
- [ ] Report drift distributions (CDF) and per-task heatmaps; compare across models/agents/decoding settings.
- [ ] Consider a z-score style outlier detector to flag paraphrases that induce unusually large behavioral changes.

## Quotes / details to potentially cite

- Abstract-level framing: they “investigate how LLMs respond to prompts that differ in their token-level realization but preserve the same semantic intent,” and propose PBSS “for measuring behavioral drift … under semantically equivalent prompt rewordings.”
- PBSS definition (paraphrased): compute cosine distance between sentence-encoder embeddings of model outputs across prompt paraphrases; use the resulting drift matrix/CDF to characterize QoS instability.
