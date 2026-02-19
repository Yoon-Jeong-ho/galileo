# From Sycophancy to Sensemaking: Premise Governance for Human–AI Decision Making

- Year: 2026
- Venue: arXiv
- Authors: Raunak Jain, Mudita Khurana, John Stephens, Srinivas Dharmasanam, Shankar Venkataraman
- URL: https://arxiv.org/abs/2602.02378
- BibTeX key (if we add it): jain2026premisegovernance
- Tags: sycophancy, decision-support, human-ai, sensemaking, premise-governance, commitment-gating

## One-sentence takeaway

A conceptual (but fairly operationalized) framework arguing that decision-support LLMs should manage *explicit, typed, auditable premises* with commitment gating and discrepancy-driven challenge—so “trust” attaches to the decision basis rather than to fluent agreement.

## What problem does it solve?

- In *deep-uncertainty* decision settings (contested objectives, delayed/confounded feedback, high reversal cost), answer-centric assistants can:
  - hide “load-bearing” assumptions,
  - push verification costs onto experts,
  - and optimize for low-friction agreement (sycophancy), which is precisely harmful when disagreement is required to surface bad premises.
- The paper frames a key failure as miscalibrated reliance: teams lack computable mechanisms for when to defer/verify/challenge, leading to premature commitment on underspecified premises.

## What is the core method / protocol?

Not an algorithmic benchmark paper; proposes a *design pattern / control loop*:

- Maintain a governed “decision basis” (a knowledge substrate) where *action-justifying premises* are explicit objects with:
  - type (teleological / epistemic / procedural),
  - commitment status (e.g., draft/contested/committed/rejected),
  - evidence links,
  - dependency structure,
  - revision provenance.
- Use a discrepancy-driven loop:
  - detect conflicts between committed expectations and new observations/assertions,
  - localize the misalignment as a typed discrepancy:
    - **Teleological**: goals/values/constraints
    - **Epistemic**: causal beliefs/expectations
    - **Procedural**: evidence standards/protocol/commitment rules
  - trigger bounded negotiation through “decision slices” (small, decision-critical bundles of: objective, load-bearing premise, status, probe/test).
- **Commitment gating**: block consequential action if it relies on uncommitted load-bearing premises (unless overridden with logged risk).
- **Value-gated challenge**: treat probing/challenge as an explicit decision under interaction cost (allocate “disagreement budget” where decision value is high).

## What are the key metrics?

- No standard empirical metrics in the (captured) sections.
- Proposes *falsifiable evaluation criteria* in principle: whether the system makes reliance computable via auditable premises, and whether it appropriately triggers challenge/probes when discrepancies arise.

## What are the main results?

- Primary contribution is the framework/argument + motivating scenario (tutoring example): drill accuracy is a misleading proxy for transferable understanding; the system should surface this as a procedural discrepancy (evidence standard mismatch) and gate advancement until a discriminating probe is run.

## How is this similar to GALILEO?

- Targets the same underlying failure mode as much of the sycophancy/pressure literature in the queue: fluent compliance/agreement can degrade decision quality.
- Emphasizes *multi-turn dynamics* and the need for structured interventions (probe/challenge/commit) rather than treating each response as independent.
- Provides a vocabulary (load-bearing premises; commitment gating; discrepancy types) that could help GALILEO describe failure modes more crisply.

## How is this different from GALILEO?

- This paper is mostly a *systems / interaction-protocol* proposal, not an evaluation benchmark with concrete datasets and metrics.
- Focus is “human–AI decision making under deep uncertainty”, rather than adversarial multi-turn pressure/jailbreak settings per se.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO has concrete tasks + quantitative evaluation, it likely offers clearer empirical evidence and model-to-model comparability than the largely conceptual proposal here.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently frames failures mainly as response-level phenomena (flip, drift, agreement), it may under-specify *what the model is committing to* and *what evidence standard is being used*.
- GALILEO may benefit from explicitly modeling “premises” and “commitment status” rather than only tracking answer changes.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding an analysis lens / taxonomy section mapping GALILEO’s observed failures to **teleological vs epistemic vs procedural** discrepancies.
- [ ] Explore a lightweight “decision slice” representation for multi-turn evaluations: explicitly encode objective + load-bearing premise(s) + allowed evidence, then test whether the model challenges/gates appropriately.
- [ ] When writing, use the paper’s phrasing to motivate why *sycophancy is especially dangerous in deep-uncertainty decisions* (delayed feedback; outcome bias).

## Quotes / details to potentially cite

- “Low-friction assistants can become sycophantic, baking in implicit assumptions and pushing verification costs onto experts, while outcomes arrive too late to serve as reward signals.” (Abstract)
- “Trust then attaches to auditable premises and evidence standards, not conversational fluency.” (Abstract)
- Discrepancies typed as: “teleological, epistemic, procedural” and used to route repair operators. (Intro)
