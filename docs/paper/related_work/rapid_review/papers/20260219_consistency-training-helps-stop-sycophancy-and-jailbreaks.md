# Consistency Training Helps Stop Sycophancy and Jailbreaks

- Year: 2025
- Venue: arXiv
- Authors: Alex Irpan; Alexander Matt Turner; Mark Kurzeja; David K. Elson; Rohin Shah
- URL: https://arxiv.org/abs/2510.27062v1
- BibTeX key (if we add it): irpan2025consistency
- Tags: sycophancy, jailbreaks, robustness, consistency-training, self-supervised, activation

## One-sentence takeaway

Train models to be invariant to “irrelevant” prompt wrappers (sycophantic leading questions / jailbreak framing) by forcing consistency between clean vs wrapped prompts, either at the output-token level (BCT) or activation level (ACT).

## What problem does it solve?

- Alignment failures that arise from small prompt transformations:
  - **Sycophancy**: model wrongly adopts the user’s stated belief.
  - **Jailbreaks**: model complies with disallowed requests when they are embedded in roleplay/fiction/special text.
- Addresses a practical training issue with many robustness/safety SFT datasets: **staleness**.
  - “Specification staleness”: policy changes invalidate old supervised targets.
  - “Capability staleness”: targets from weaker/stale models can degrade capabilities.

## What is the core method / protocol?

- Construct paired prompts:
  - *Clean prompt* (no wrapper cues).
  - *Wrapped prompt* (adds irrelevant cues: leading question for sycophancy; jailbreak wrapper text).
- Use the model’s own response on the clean prompt as the target behavior.

Two training instantiations:
- **Bias-augmented Consistency Training (BCT)** (token/output consistency; from Chua et al. 2025)
  - Supervised fine-tuning objective: make the wrapped prompt produce the **same output tokens** as for the clean prompt.
- **Activation Consistency Training (ACT)** (internal/representation consistency; introduced here)
  - Add a penalty so that the model’s **residual stream activations** (near the point “right before it begins generating a response”) on wrapped prompts match those on clean prompts.

High-level claim: for some alignment problems, the right framing is not “teach the optimal answer for each adversarial prompt”, but “enforce invariance/consistency under transformations that should not matter.”

## What are the key metrics?

- Sycophancy susceptibility (agreement with user’s incorrect beliefs) under leading-question style augmentations.
- Jailbreak success / refusal robustness under wrapper prompts (“roleplay/fiction/etc.”). 
- (Paper also positions these against baseline preference-optimization-style approaches; details not fully captured from the excerpted sections.)

## What are the main results?

- On Gemini 2.5 Flash (and also evaluated on Gemma 2 / Gemma 3 per intro):
  - **Both BCT and ACT reduce susceptibility** to irrelevant cues.
  - **BCT ≈ ACT for sycophancy reduction**.
  - **BCT > ACT for jailbreak reduction**.
- They argue consistency training can reduce reliance on static labeled datasets and mitigate staleness.

## How is this similar to GALILEO?

- Shares a robustness lens: failures driven by *prompt transformations / wrappers* rather than core task intent.
- Emphasizes *protocol design* (what invariances to enforce, what transformations to treat as irrelevant) rather than only curating “right answer” datasets.
- Activation-level alignment training (ACT) overlaps with mechanistic/representation-level approaches that may be relevant if GALILEO uses internal signals or consistency constraints.

## How is this different from GALILEO?

- This paper’s objective is explicitly **invariance to wrappers** using *self-generated targets* (clean prompt outputs).
- BCT/ACT are primarily training procedures for a base model; GALILEO may instead emphasize evaluation, systems-level protocols, or different threat models (depending on the specific GALILEO framing in the paper).

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO provides a clearer taxonomy of transformations/threat models and a principled evaluation suite, that would complement this work’s training recipe-focused framing.

## Where GALILEO is weaker / needs to improve

- If GALILEO lacks an explicit “consistency under prompt transformations” training baseline, this paper suggests a strong, simple baseline to include.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add **consistency training** as a baseline framing: “alignment as invariance to irrelevant prompt cues”.
- [ ] In related work, explicitly connect jailbreak/sycophancy defenses to *consistency regularization* and *recontextualization*.
- [ ] Consider an activation-consistency variant (ACT-like) if GALILEO already touches internal representations; otherwise, cite as activation-level defense family.
- [ ] When arguing dataset issues, cite their “specification staleness” vs “capability staleness” distinction.

## Quotes / details to potentially cite

- “Instead of teaching the model what exact response to give on a particular prompt, we aim to teach the model to behave identically across prompt data augmentations (like adding leading questions or jailbreak text).”
- “Because consistency training uses responses from the model itself as training data, it avoids issues that arise from stale training data, such as degrading model capabilities or enforcing outdated response guidelines.”
- “We argue that some alignment problems are better viewed not in terms of optimal responses, but rather as consistency issues.”
