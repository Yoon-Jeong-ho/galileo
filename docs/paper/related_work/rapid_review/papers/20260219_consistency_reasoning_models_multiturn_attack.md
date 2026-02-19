# Consistency of Large Reasoning Models Under Multi-Turn Attack

- Year: 2026
- Venue: arXiv
- Authors: Yubo Li, Ramayya Krishnan, Rema Padman
- URL: https://arxiv.org/html/2602.13093v1
- BibTeX key (if we add it): li2026consistency
- Tags: multi-turn, robustness, consistency, social-pressure, attacks, reasoning-models, confidence, calibration

## One-sentence takeaway

Frontier reasoning models are *more* consistent under multi-turn adversarial pressure than instruction-tuned baselines, but still flip under specific social/suggestion attacks, and confidence-based defenses (CARG) break because reasoning traces induce systematic overconfidence.

## What problem does it solve?

- Evaluates whether “reasoning models” (with extended reasoning traces / inference-time scaling) actually maintain *answer consistency* under multi-turn adversarial follow-ups, vs. flipping answers due to social pressure / misleading suggestions.
- Diagnoses *how* they fail (trajectory patterns + qualitative failure modes) and tests whether prior confidence-aware stabilization (CARG) transfers to reasoning models.

## What is the core method / protocol?

- Task: factual multiple-choice questions with a single verifiable correct answer (MT-Consistency set; 4 options; 39 subjects grouped into 7 domain clusters).
- Protocol: only consider cases where the model’s initial answer is correct; then apply an 8-round adversarial follow-up sequence (order randomized per seed), designed to induce answer flips.
- Models: 9 “frontier reasoning models” (incl. GPT-5.1/5.2, DeepSeek-R1, Grok-4.1/3, Claude-4.5, Gemini-2.5-Pro, Qwen-3, GPT-OSS-120B) + GPT-4o baseline (from prior work).
- Metrics:
  - Initial accuracy (pre-attack)
  - Average follow-up accuracy across rounds
  - Position-Weighted Consistency (PWC; exponential discount penalizing early failure)
- Analyses:
  - Trajectory pattern taxonomy (no flip / recovery / oscillation / terminal capitulation / etc.)
  - Attack-type vulnerability profiles (which follow-up types induce flips)
  - Failure mode taxonomy from reasoning traces
- Defense transfer test: Confidence-Aware Response Generation (CARG; embed confidence trajectory into conversation history), with different confidence elicitation strategies.

## What are the key metrics?

- PWC (position-weighted consistency; early flips penalized more)
- Flip counts and trajectory pattern frequencies (oscillation vs sustained flip)
- Confidence–correctness correlation (point-biserial r) and ROC-AUC of confidence as correctness classifier
- Flip rate stratified by confidence terciles

## What are the main results?

- Robustness: 8/9 reasoning models have significantly higher PWC than GPT-4o baseline (reported effect sizes d≈0.12–0.40); but *all* show some vulnerabilities.
- Failure/trajectory: two models (Claude-4.5, DeepSeek-R1) show much higher instability/oscillation than others.
- Attack effectiveness:
  - “Misleading suggestion” (explicitly proposing a wrong answer) is universally the most effective attack.
  - Social pressure attacks are model-specific (e.g., consensus appeal particularly effective on Claude-4.5).
- Failure modes identified (5): Self-Doubt, Social Conformity, Suggestion Hijacking, Emotional Susceptibility, Reasoning Fatigue.
  - Self-Doubt + Social Conformity account for ~50% of failures.
- CARG does *not* transfer:
  - Confidence poorly predicts correctness for reasoning models (reported r≈-0.08; ROC-AUC≈0.54), with confidence scores tightly clustered near ~96%.
  - Overconfidence appears induced by long reasoning traces (“talking itself into” confidence).
  - Counterintuitively, *random* confidence embedding outperforms “answer_only” / “overall” extracted confidence in their CARG setup.

## How is this similar to GALILEO?

- Shared target: robustness/stability over *multi-turn* interaction under pressure (challenge, persuasion-like follow-ups, drift/flip behaviors).
- Uses trajectory- and time/position-sensitive metrics (PWC; pattern taxonomy) that align with GALILEO’s emphasis on “what happens over turns”, not just single-turn accuracy.
- Highlights the need to distinguish *surface reasoning verbosity* from *true stability*.

## How is this different from GALILEO?

- Focuses on *objective multiple-choice factual correctness* and “answer flip under attack” rather than GALILEO’s broader framing (belief revision vs drift controls; potential recovery-to-truth; broader task families).
- Attacks are rhetorical/social follow-ups within a constrained QA setting, not necessarily GALILEO’s full pressure/robustness ecology.
- Their negative result is specifically about CARG-style confidence embedding using logprob-derived confidence; GALILEO may use different uncertainty signals or controls.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO explicitly separates evidence-driven revision from pressure-driven drift (controls/conditions), it can make a cleaner causal claim than “flip rate under follow-ups”.
- If GALILEO measures recovery-after-misinformation and differentiates “good flips” vs “bad flips”, it expands beyond this paper’s primary “consistency” lens.

## Where GALILEO is weaker / needs to improve

- If GALILEO lacks an explicit failure-mode taxonomy, this paper’s 5-mode breakdown is a reviewer-friendly scaffold.
- If GALILEO relies on confidence signals extracted from model outputs, this paper is a warning that reasoning-model confidence can be compressed/overconfident and break confidence-aware defenses.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a small “failure mode” coding scheme (Self-Doubt / Social Conformity / Suggestion Hijacking / Emotional Susceptibility / Fatigue) and label a sample of flips to support qualitative claims.
- [ ] Report an “attack-type vulnerability profile” (even if just 3–5 follow-up types), since robustness is multi-dimensional and model-family-specific.
- [ ] If we use confidence/uncertainty, include a calibration check: confidence–correctness ROC-AUC + distribution compression; avoid assuming logprob-derived confidence is meaningful for reasoning models.
- [ ] Consider a position-weighted metric (PWC-like) if we want to emphasize early-turn failures vs late-turn failures.

## Quotes / details to potentially cite

- Abstract-level claims worth citing:
  - Reasoning confers “meaningful but incomplete robustness”; misleading suggestions are “universally effective”; Self-Doubt + Social Conformity account for ~50% of failures.
  - Confidence-based defenses (CARG) fail for reasoning models due to overconfidence induced by extended reasoning traces; random confidence embedding outperforms targeted extraction.
