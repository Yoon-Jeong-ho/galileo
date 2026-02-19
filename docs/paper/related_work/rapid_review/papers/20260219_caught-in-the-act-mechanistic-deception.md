# Caught in the Act: a mechanistic approach to detecting deception

- Year: 2025
- Venue: arXiv
- Authors: Gerard Boxo; Ryan Socha; Daniel Yoo; Shivam Raval
- URL: https://arxiv.org/abs/2508.19505
- BibTeX key (if we add it): boxo2025caught
- Tags: deception, mechanistic, linear-probes, white-box-eval, monitoring

## One-sentence takeaway

Linear probes over middle-layer residual activations can detect whether an LLM is producing *deceptive* (vs non-deceptive) arguments on a synthetic factual-QA setup, with accuracy rising sharply with model scale and being strongest for “reasoning” variants.

## What problem does it solve?

- If deception is a salient misalignment-related failure mode, we may want *instrumentation* that flags deceptive generations (a “check engine light”), rather than relying only on output-based detectors.
- They study whether internal activations contain linearly-decodable information that distinguishes deceptive vs non-deceptive responses.

## What is the core method / protocol?

- Build a *synthetic* dataset of (question, choices, answer-direction) prompts seeded from MMLU, then have target models generate *arguments* that are either deceptive or non-deceptive.
- Collect hidden activations (they focus on the final token position) across layers for each generated argument.
- Train a linear classifier (logistic regression probe) to predict deceptive vs non-deceptive from activations.
- Compare to a black-box baseline: “LLM-as-judge” text classification (GPT-4o / GPT-4o-mini) that labels deception from the argument text.
- Reliability / structure checks:
  - INLP (iterative nullspace projection) to estimate how many independent linear directions encode deception.
  - Saliency-style attribution (gradient-based) to interpret what drives the probe.

## What are the key metrics?

- Deception-vs-non-deception classification accuracy of probes (layer-wise).
- Trends vs model size (1.5B/3B/7B/14B) and vs “reasoning” distill variants.
- Number of INLP rounds until accuracy drops to chance (proxy for how many “deception directions” exist).

## What are the main results?

- Probes can reach >90% accuracy on larger models / reasoning variants (reported for Llama/Qwen families; includes DeepSeek-r1 distilled variants).
- Strong model-size trend:
  - ~chance at 1.5B; ~70–80% for >=7B non-reasoning; >90% for reasoning variants.
- Layer-wise pattern:
  - near-random in early layers, peaks in mid layers (~two-thirds depth), slightly declines late.
- Many linear directions encode deception (via INLP): on the order of ~20 (Qwen 3B) up to ~100 (DeepSeek 7B / Qwen 14B) before performance collapses to chance.

## How is this similar to GALILEO?

- Supports the broader “multi-turn / behavioral” robustness agenda by arguing for *mechanistic* monitors rather than purely outcome-based evaluation.
- Uses *directionality* / representation-space diagnostics (linear separability, subspaces) that can inspire how we analyze preference/pressure-induced behavioral shifts.

## How is this different from GALILEO?

- Focuses on deception detection on a synthetic factual QA argument-generation setup, not multi-turn conversational drift/pressure or belief revision.
- Assumes white-box access to activations; many GALILEO-style evaluations may be black-box or interaction-based.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO emphasizes *interactional* robustness (multi-turn, pressure, persuasion), it may better match real deployment dynamics than synthetic single-episode argument generation.

## Where GALILEO is weaker / needs to improve

- If GALILEO is currently mostly black-box, this paper suggests white-box “check engine light” style monitors could materially improve detection/diagnosis of undesirable behaviors.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding a short related-work paragraph framing *representation-space monitors* (linear probes / activation-based indicators) as complementary to interaction-based robustness benchmarks.
- [ ] If we have access to internal activations for any target models, consider a small pilot: can we linearly predict *pressure-induced flips* / *agreement drift* from mid-layer activations? (Even if only as an analysis experiment.)
- [ ] When discussing “behavioral failures,” note that detectability can be layer-dependent (early random, mid-layer peak), motivating where to probe.

## Quotes / details to potentially cite

- “Linear probes on LLMs internal activations can detect deception … with extremely high accuracy … >90% … for reasoning counterparts.” (Abstract)
- “Layer-wise … three-stage pattern: near-random (50%) in early layers, peaking in middle layers, and slightly declining in later layers.” (Abstract)
- “Iterative null space projection … multitudes of linear directions that encode deception … ~20 … to nearly 100 …” (Abstract)
