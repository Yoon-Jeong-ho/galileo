# LLM-Generated Ads: From Personalization Parity to Persuasion Superiority

- Year: 2025
- Venue: arXiv
- Authors: Elyas Meguellati, Stefano Civelli, Lei Han, Abraham Bernstein, Shazia Sadiq, Gianluca Demartini
- URL: https://arxiv.org/abs/2512.03373
- BibTeX key (if we add it): meguellati2025llm_generated_ads
- Tags: persuasion, advertising, human-eval, personalization

## One-sentence takeaway

Human-subject studies suggest LLM-written ads match human experts on trait-personalized ads but outperform humans when leveraging classic persuasion principles (authority/consensus/etc.), even with an “AI-detection” penalty.

## What problem does it solve?

- Quantifies whether LLMs can generate *effective* advertising copy (and associated narratives) compared to human experts, under two paradigms:
  - (i) personalization to personality traits, and
  - (ii) general persuasion principles.
- Addresses whether knowing content is AI-authored reduces its persuasive impact (“algorithm aversion”).

## What is the core method / protocol?

- Two randomized human-preference studies comparing AI-generated vs human-expert-created ads.
- Study 1 (personality personalization):
  - Participants: n=400.
  - Ads targeted to specific Big Five traits (focus on openness + neuroticism).
  - Tests “match” conditions (ad tailored to participant trait) and compares AI vs human expert performance.
- Study 2 (persuasion principles):
  - Participants: n=800.
  - Ads constructed using four psychological influence principles: authority, consensus, cognition (processing/fluency framing), and scarcity.
  - Measures preference for AI vs human ads, and analyzes reasons/qualitative differences.
- Additional analysis on origin awareness:
  - Looks at preference conditional on participants correctly identifying AI origin; reports an effective “detection penalty,” and a subgroup that still chooses AI even when they know.

## What are the key metrics?

- Human preference rate (AI vs human) for ads in forced-choice or comparative judgments.
- Per-principle preference rates (authority/consensus/cognition/scarcity).
- “Detection penalty” adjustment when participants correctly identify AI origin.

## What are the main results?

- Study 1: AI ads are statistically on-par with human expert ads overall.
  - Reported preference: 51.1% (AI) vs 48.9% (human), p > 0.05.
  - No significant advantage for trait-matched personality targeting (for openness/neuroticism) vs human experts.
- Study 2: AI ads significantly outperform human experts on persuasion-principle ads.
  - Overall preference: 59.1% (AI) vs 40.9% (human), p < 0.001.
  - Strongest appeals: authority (63.0%) and consensus (62.5%).
- Robustness to origin awareness:
  - Even after applying a reported 21.2 percentage-point “detection penalty” (when AI origin is correctly identified), AI ads still outperform.
  - 29.4% of participants chose AI content despite knowing it was AI-generated.
- Claimed qualitative mechanism: AI produces more “aspirational” messaging and better visual–narrative coherence.

## How is this similar to GALILEO?

- Shares a focus on *persuasion* as an evaluation target, and on quantifying behavioral effects with human-facing outcomes.
- Highlights that model behavior can be systematically analyzed across “conditions” (here: ad strategy families), similar in spirit to robustness protocols that sweep perturbation families.

## How is this different from GALILEO?

- Single-shot persuasion effectiveness in advertising, not multi-turn interaction robustness.
- Outcome is immediate *human preference* for content, not longitudinal stability, time-to-failure, or recovery trajectories.
- Does not propose a general-purpose robustness metric/protocol for conversational drift or attack resistance.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO targets multi-turn dynamics, it can measure *temporal failure processes* (e.g., hazard/turn-of-failure), not just aggregate preference.
- Can isolate conversational mechanisms (pressure, misleading followups, repeated challenges) that are closer to real multi-turn reliability/safety concerns than ad-copy comparisons.

## Where GALILEO is weaker / needs to improve

- If GALILEO lacks human-subject preference or effectiveness measures, this paper is a reminder that *human impact* endpoints matter and can diverge from automated metrics.
- Might motivate adding “origin awareness / disclosure” conditions (participants/users know the model is an LLM) as a robustness factor.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider a small “persuasion effectiveness” section in related work that distinguishes:
  - personalization (trait-targeting) vs
  - universal persuasion principles.
- [ ] In GALILEO’s evaluation design, consider an “origin disclosure” ablation (user knows model is AI / system discloses) as a covariate affecting compliance/flip rates.

## Quotes / details to potentially cite

- “LLM-generated ads achieved statistical parity with human-written ads (51.1% vs. 48.9%, p > 0.05).” (Study 1)
- “AI-generated ads significantly outperformed human-created content, achieving a 59.1% preference rate (vs. 40.9%, p < 0.001) … authority (63.0%) and consensus (62.5%).” (Study 2)
- “Even after applying a 21.2 percentage point detection penalty … AI ads still outperformed … and 29.4% … chose AI content despite knowing its origin.”