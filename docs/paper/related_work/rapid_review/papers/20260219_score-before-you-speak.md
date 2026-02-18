# Score Before You Speak: Improving Persona Consistency in Dialogue Generation using Response Quality Scores

- Year: 2025
- Venue: ECAI 2025 (camera-ready; arXiv)
- Authors: Jonathan C. Darling; Vania Dimitrova; Duygu Sarikaya; David C. Hogg
- URL: https://arxiv.org/abs/2508.06886
- BibTeX key (if we add it): score_before_you_speak_2025
- Tags: persona, consistency, multi-turn, dialogue, data-augmentation, score-conditioning

## One-sentence takeaway

SBS trains a persona-dialogue model in a single MLE-style objective by augmenting responses (noun substitutions) and conditioning training on a semantic-similarity-derived quality score, improving persona consistency on Persona-Chat / ConvAI2.

## What problem does it solve?

- Persona-based dialogue datasets are relatively small and low-diversity, making it hard to train models that stay persona-consistent over multi-turn conversations.
- Prior approaches often use multi-stage pipelines (SFT + preference/alignment stage via RL/contrastive/etc.), which adds complexity and compute.

## What is the core method / protocol?

- SBS (Score-Before-Speaking): unify response learning + “quality alignment” in *one* training setup.
- Data augmentation: part-of-speech tag responses; mask nouns and regenerate them (one at a time) to create “corrupted” (augmented) responses.
- Quality scoring: assign each augmented response a scalar score based on semantic similarity to the original (gold) response (original responses receive score 1.0).
- Training: include the score in the input prompt and train a decoder model with an MLE objective to learn a mapping between context/persona → response *conditioned on desired quality*.
- Inference: leverage the learned score-conditioning to produce more persona-consistent responses.

## What are the key metrics?

- Automatic + human evaluation on PERSONA-CHAT and ConvAI2.
- Emphasis: persona consistency / persona fidelity (plus general response quality measures).

## What are the main results?

- Reported improvements over prior baselines across both smaller (million-scale) and larger (billion-scale) dialogue models.
- Ablations suggest that *including the score in the input prompt during training* is better than conventional setups that do not expose the score to the model.

## How is this similar to GALILEO?

- Both focus on *multi-turn consistency* phenomena (here: staying consistent with a persona; GALILEO: staying consistent with ground-truth under pressure).
- Both implicitly treat failures as trajectory phenomena (consistency over turns), not just single-turn accuracy.

## How is this different from GALILEO?

- SBS is about *persona-conditioned generation quality* (benign dialogue personalization), not truth maintenance under adversarial/persuasive pressure.
- SBS addresses limited data + training objective design; GALILEO is primarily an evaluation protocol (survival/TOF/recovery + neutral re-asking control) on ground-truth tasks.

## Where GALILEO is stronger / cleaner (if true)

- GALILEO’s SSOT-style protocol explicitly separates pressure effects from drift (neutral control) and provides survival/TOF/recovery metrics tied to *verifiable ground-truth*.
- SBS uses semantic similarity as a proxy for “quality,” which may conflate style/semantic closeness with correctness/consistency.

## Where GALILEO is weaker / needs to improve

- GALILEO currently targets truth/answer stability; it may under-cover “persona/role-play consistency” as a distinct axis of multi-turn instability.
- GALILEO may benefit from adopting the *single-objective, score-conditioned* framing when discussing interventions/mitigations (even if not adopting this exact method).

## Action items for GALILEO (experiments / method / writing)

- [ ] Related work writing: add a short paragraph contrasting (i) persona-consistency training/augmentation lines (SBS) vs (ii) pressure-induced truth decay / sycophancy evaluation (GALILEO), emphasizing that GALILEO’s target is ground-truth robustness.
- [ ] Consider a small “persona-consistency” sidebar in taxonomy: persona drift vs truth drift (and why we focus on ground-truth tasks).

## Quotes / details to potentially cite

- “SBS (Score-Before-Speaking) … unifies the learning of responses and their relative quality into a single step.”
- “We … mask and regenerate nouns … [and] score the augmented responses based on semantic closeness to the original … [and] incorporate these scores … during training …”
- Datasets used: PERSONA-CHAT and ConvAI2.
