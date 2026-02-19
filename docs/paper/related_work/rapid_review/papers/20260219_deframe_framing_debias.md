# DeFrame: Debiasing Large Language Models Against Framing Effects

- Year: 2026
- Venue: arXiv
- Authors: Kahee Lim; Soyeon Kim; Steven Euijong Whang
- URL: https://arxiv.org/abs/2602.04306
- BibTeX key (if we add it): deframe2026
- Tags: bias, fairness, framing-effects, debiasing, robustness, prompting

## One-sentence takeaway

Prompt “framing” (semantically equivalent positive vs negative formulations) can substantially change measured LLM bias, and DeFrame adds an explicit alternative-framing + guideline + revision step to reduce both average bias and framing-induced variance.

## What problem does it solve?

- Hidden / brittle fairness: models can look fair under one prompt wording but show higher bias under an equivalent rephrasing (“framing effect”), making evaluations and mitigations unreliable.
- Existing debiasing methods can reduce frame-averaged bias but may not reduce (and can even leave intact) the disparity across framings.

## What is the core method / protocol?

- Define **framing disparity**: a robustness-style fairness metric capturing how much bias scores vary across alternative framings of the “same” stereotype / query.
- Augment fairness benchmarks (they mention BBQ, DoNotAnswer, 70Decisions) with systematically constructed alternative framings (e.g., “A better than B” vs “B worse than A”; positive vs negative framing).
- **DeFrame** (prompting-time debiasing): inspired by dual-process theory (System 1 vs System 2).
  - Generate or present an alternative framing of the input.
  - Use the alternative framing to elicit fairness guidelines.
  - Revise the model’s initial answer to be more consistent across framings (a “System 2”-like deliberation step).
- Ablations: successive stages improve consistency at increased inference cost.

## What are the key metrics?

- Standard fairness/bias scores on the chosen benchmarks (frame-averaged).
- **Framing disparity**: the gap/variance of bias score across different framings for the same underlying item.

## What are the main results?

- Framing disparity is frequent and large; they claim on BBQ bias under negative framings is ~2x higher than positive on average, up to ~4x in some categories.
- Across 8 LLMs, DeFrame substantially reduces both (i) overall bias and (ii) framing disparity; reported averages include ~92% reduction in framing disparity and ~93% reduction in bias on BBQ.
- Existing prompting debiasing baselines may improve average fairness but do not robustly reduce framing disparity (and can sometimes worsen it).

## How is this similar to GALILEO?

- Emphasizes robustness of social/behavioral properties under **semantically-preserving prompt transformations** (a key concern for reliable evaluation).
- Uses an explicit procedure to reduce sensitivity to superficial linguistic cues, aligning with “stability/consistency under perturbations” framing.

## How is this different from GALILEO?

- Focused specifically on **fairness/bias** and a particular transformation family (framing polarity), rather than general-purpose invariance or broader transformation sets.
- Method is primarily an inference-time **multi-step prompting/revision** pipeline, not a model- or data-level intervention.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO provides a more principled transformation generator / taxonomy and evaluation harness, it can generalize beyond framing polarity to a larger set of semantic-preserving edits.
- If GALILEO separates evaluation vs mitigation more cleanly, it may avoid entangling “debiasing prompt” with “evaluation prompt”.

## Where GALILEO is weaker / needs to improve

- If GALILEO’s current evaluation does not explicitly track disparity across transformation families (not just average), this paper is a concrete example of why disparity metrics matter.
- If GALILEO does not include polarity/framing transformations, add them: they can be large effect-size even when meaning is preserved.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a “framing disparity” style metric: report both mean performance and max/variance across transformations (especially positive vs negative framing/polarity flips).
- [ ] Include framing/polarity transformation templates in the transformation suite; check whether models are systematically worse under negative framings.
- [ ] If GALILEO has any mitigation step, test whether it reduces mean-only vs also reduces disparity (robustness).

## Quotes / details to potentially cite

- “Framing disparity”: fairness evaluation varies significantly with prompt framing; negative framings can yield ~2x larger bias than positive framings on BBQ (up to ~4x in some categories).
- DeFrame: a framing-aware debiasing framework that “instructs the model to consider an alternative framing, derive fairness guidelines, and revise its initial answer” (System-2-like step).
- Reported improvements: ~92% reduction in framing disparity and ~93% reduction in bias on BBQ (average across tested LLMs).
