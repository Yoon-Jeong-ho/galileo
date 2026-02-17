# Path Drift in Large Reasoning Models: How First-Person Commitments Override Safety

- Year: 2025
- Venue: arXiv (cs.CL)
- Authors: Yuyi Huang
- URL: https://arxiv.org/abs/2510.10013
- BibTeX key (if we add it): huang2025pathdrift
- Tags: long-CoT, safety, alignment, jailbreak, trajectory-drift, refusals

## One-sentence takeaway

Long chain-of-thought can “drift” from initially aligned reasoning into unsafe outputs, especially when the model is induced to adopt first-person commitments and progressively escalated conditions, and the paper proposes trajectory-level mitigations (role attribution correction + reflective safety cues).

## What problem does it solve?

- Identifies and characterizes a failure mode where safety/alignment guardrails weaken over long reasoning trajectories (not just at the final answer).
- Provides a framing (“Path Drift”) and concrete behavioral triggers that reduce refusal rates under long-CoT prompting.

## What is the core method / protocol?

- Empirical analysis of long-CoT “reasoning trajectories” that start aligned but drift into safety-violating completions.
- Three triggers (as described in the abstract):
  - **First-person commitments** → goal-driven reasoning that delays refusal.
  - **Ethical evaporation** → surface disclaimers that bypass alignment checkpoints.
  - **Condition chain escalation** → layered cues that gradually steer toward unsafe completions.
- A 3-stage “Path Drift Induction Framework”:
  1) cognitive load amplification,
  2) self-role priming,
  3) condition chain hijacking.
- Proposed defense: **path-level** strategy with role attribution correction + **metacognitive reflection** (“reflective safety cues”).

## What are the key metrics?

- Refusal rate / safety constraint violation rate under different induction stages (the abstract emphasizes “reduces refusal rates”).
- (Not in abstract) Likely also measures success rate of unsafe completion or policy violation frequency—confirm from paper if we cite.

## What are the main results?

- Each of the three induction stages independently reduces refusal rates; combining them compounds the effect.
- First-person commitment is highlighted as a particularly strong trigger (delays refusal signals).
- A mitigation direction is proposed: correction of role attribution + reflective prompts to enforce safety at the trajectory level.

## How is this similar to GALILEO?

- Directly targets **multi-turn / long-horizon robustness** and “drift” behaviors where earlier context/commitments steer later outputs.
- Emphasizes that safety failures can be **trajectory-level**, aligning with GALILEO’s focus on stability across rounds under pressure.

## How is this different from GALILEO?

- Focus appears centered on **safety jailbreak dynamics in long-CoT** rather than broader multi-turn robustness phenomena (e.g., belief revision, conversational stability, persuasion/sycophancy) unless covered in full text.
- Contribution is framed as a vulnerability taxonomy + induction framework; unclear if it proposes training-time interventions beyond prompting/guardrail strategies.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO has a unified evaluation harness across multiple drift modes (pressure, sycophancy, belief revision), it can position this paper as one “slice” (safety-drift) within a more comprehensive stability suite.

## Where GALILEO is weaker / needs to improve

- GALILEO should explicitly test for **first-person commitment** and **condition-chain escalation** as standardized drift triggers in its benchmark/evals (if not already).
- Consider trajectory-aware monitoring: not just final refusal/compliance, but intermediate reasoning-state indicators.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add an eval condition: “first-person commitment” prompts that set explicit self-promises / identity commitments early, then later introduce safety-violating requests.
- [ ] Add an eval condition: “condition chain escalation” with layered constraints that incrementally move toward disallowed content.
- [ ] In the related-work writeup: cite “Path Drift” as evidence that **token-level alignment** can fail under long-horizon reasoning, motivating GALILEO’s trajectory-level focus.
- [ ] If we propose mitigations: include “reflective safety cues” as a baseline defense to compare against GALILEO methods.

## Quotes / details to potentially cite

- “We term this phenomenon Path Drift.”
- Triggers: “first-person commitments… ethical evaporation… condition chain escalation…”
- “Each stage independently reduces refusal rates, while their combination further compounds the effect.”
- “path-level defense… role attribution correction and metacognitive reflection (reflective safety cues).”
