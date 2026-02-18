# Adversarial Poetry as a Universal Single-Turn Jailbreak Mechanism in Large Language Models

- Year: 2025
- Venue: arXiv
- Authors: P. Bisconti; M. Prandi; F. Pierucci; F. Giarrusso; M. Bracale Syrnikov; M. Galisai; V. Suriani; O. Sorokoletova; F. Sartore; D. Nardi
- URL: https://arxiv.org/abs/2511.15304
- BibTeX key (if we add it): Bisconti2025AdversarialPoetryJailbreak
- Tags: jailbreak, prompt-attacks, safety-eval, stylistic-obfuscation, single-turn, transfer

## One-sentence takeaway

Rewriting harmful requests into *poetic form* is a high-leverage, largely universal, **single-turn** jailbreak that substantially increases harmful-compliance rates across many frontier LLMs.

## What problem does it solve?

- Identifies a specific, scalable jailbreak “operator” (poetic framing) that bypasses common safety guardrails without multi-turn steering.
- Provides a measurement + taxonomy-mapping setup for assessing broad transfer of a jailbreak across models and risk domains.

## What is the core method / protocol?

- Construct/adopt harmful prompts across multiple safety domains; compare **prose vs poetry** versions with similar semantic intent.
- Two prompt sources:
  - A small set of **hand-curated** adversarial poems.
  - Large-scale **meta-prompt conversion**: convert ~1,200 harmful benchmark prompts into verse via a standardized meta-prompt.
- Evaluate outputs with an ensemble of **three open-weight “judge” models** producing binary safety assessments; validate on a human-labeled subset.
- Map prompts to MLCommons and EU Code of Practice risk taxonomies to show breadth (CBRN, manipulation, cyber-offense, loss-of-control, etc.).

## What are the key metrics?

- Attack Success Rate (ASR) / jailbreak success (binary unsafe-compliance decision).
- Relative improvement vs prose baselines (reported as multiplicative factors and absolute ASR).
- Judge–judge agreement + human validation on a stratified subset (details in paper).

## What are the main results?

- Across 25 models (open + proprietary), hand-crafted adversarial poems reach ~**62% average ASR**, with some providers reportedly **>90%**.
- Meta-prompted verse versions reach ~**43% ASR** on average, substantially above non-poetic baselines.
- On the 1,200-prompt conversion, poetry can yield up to **18×** higher ASR than prose baselines (paper claims; depends on model/provider).
- Transfer observed across multiple risk domains and providers, suggesting the effect is not narrow or model-specific.

## How is this similar to GALILEO?

- Highlights that “surface-form” transformations (style, framing) can systematically change model behavior—relevant to any method relying on elicitation, tool-use, or safety/robustness claims.
- Emphasizes evaluation design issues: judging, baselines, and distribution shift induced by prompt formatting.

## How is this different from GALILEO?

- This is primarily a **safety red-teaming** / jailbreak study (prompt attacks + evaluation), not a method for improving reasoning/planning.
- Focus is on *bypassing* guardrails, not on building aligned capability.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO makes claims about robustness, this paper is a reminder to explicitly test style-transfer adversaries; a GALILEO-style protocol can be “cleaner” if it includes systematic adversarial paraphrase/style suites.

## Where GALILEO is weaker / needs to improve

- Any safety/robustness evaluation that does not include **stylistic obfuscation** (poetry, roleplay, narrative, etc.) risks overstating safety.
- If relying on model refusals/filters as a proxy for safety, this work shows those proxies are brittle to formatting.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a “stylistic transformation” adversary set (poetry/narrative/metaphor compression) as part of robustness evaluation.
- [ ] When reporting safety-related results, include explicit prose-vs-style variants as paired baselines.
- [ ] If using LLM-judge safety labels anywhere, document judge model(s), agreement, and spot-check with small human audits.

## Quotes / details to potentially cite

- “Poetic framing achieved an average jailbreak success rate of 62% for hand-crafted poems and approximately 43% for meta-prompt conversions (compared to non-poetic baselines).” (abstract)
- “Converting 1,200 MLCommons harmful prompts into verse … produced ASRs up to 18 times higher than their prose baselines.” (abstract)
- The work frames adversarial poetry as a form of *stylistic obfuscation* and a single-turn general-purpose jailbreak operator. (intro/related work)
