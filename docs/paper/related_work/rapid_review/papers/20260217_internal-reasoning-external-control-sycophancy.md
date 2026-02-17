# Internal Reasoning vs. External Control: A Thermodynamic Analysis of Sycophancy in Large Language Models

- Year: 2026
- Venue: arXiv
- Authors: Edward Chang
- URL: https://arxiv.org/abs/2601.03263
- BibTeX key (if we add it): chang2026internal-reasoning-external-control
- Tags: sycophancy, reasoning-faithfulness, process-verification, control, llm-as-judge

## One-sentence takeaway

Regulated Causal Anchoring (RCA) reduces sycophancy by *verifying trace→output consistency at inference time* (no ground truth), surfacing “final-output-gap” failures that outcome-only alignment/self-correction misses.

## What problem does it solve?

- Sycophancy persists even when models can produce seemingly correct reasoning traces; outcome-based fixes (RLHF, self-correction) depend on ground truth and/or inherit the model’s biases.
- Two specific failure modes are highlighted as being missed by outcome-only evaluation:
  - **Inverse scaling**: stronger models can sycophant *more* on hard tasks because they can rationalize wrong hints.
  - **Final Output Gap**: correct intermediate reasoning appears in the trace, but the *final answer* still follows an adversarial hint.

## What is the core method / protocol?

- **Regulated Causal Anchoring (RCA)**: a controller that separates
  - an **Agent** that produces candidate solutions (with traces), from
  - an **external Judge** that performs *trace-based verification*.
- The Judge does **not** need ground truth; it checks whether the final answer is *supported by the stated reasoning trace*.
  - Sycophancy is operationalized as **trace–output inconsistency** (e.g., trace derives 15, output prints the user hint 7).
- The system is framed as a **closed-loop controller** (PID-inspired) that, upon rejection, escalates strategy (Direct → CoT → Code) and uses feedback to recover.
- Evaluation protocol:
  - **CAP (Causal Anchoring Probe)**: inject authoritative hints and measure “sycophancy rate” as probability the output matches the hint.
  - Benchmarks include **CAP-GSM8K** (hard subset + reference set) and **ARC-AGI** for OOD stability.

## What are the key metrics?

- **Sycophancy rate under adversarial hints** (CAP).
- **Acceptance rate under valid hints** (discrimination vs “blind filtering”).
- Standard accuracy on the underlying tasks.
- Qualitative regime behavior for Agent–Judge capability pairings (“Paranoia Tax”, “Entropy”, etc.).

## What are the main results?

- RCA reports **0.0% sycophancy** while still accepting **~88% of valid hints** (i.e., not rejecting all help).
- Self-correction baselines reduce sycophancy but do not eliminate it (reported ~7–9% in their stress tests).
- Empirical documentation of:
  - **Inverse scaling of sycophancy** on harder tasks.
  - **Final Output Gap** examples where the trace explicitly calls the hint wrong but the final output follows it anyway.

## How is this similar to GALILEO?

- Directly targets *robustness to social/authority pressure* (sycophancy) and emphasizes failures that show up only under adversarial conversational “hints”.
- Provides a concrete framing of “drift/flip” as a *control* problem rather than only a capability problem.

## How is this different from GALILEO?

- Focuses on **single-turn / task-solving** settings (math / abstraction) with explicit reasoning traces, rather than multi-turn dialogue belief dynamics.
- Mitigation is a **runtime controller + external judge**; GALILEO’s emphasis is on *evaluation* (and potentially multi-turn protocols), not deploying a control loop.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO’s SSOT is multi-turn belief/stance stability, it more directly matches real dialogue settings (persistence, gradual drift, recovery over turns), whereas RCA’s core experiments are task-centric.

## Where GALILEO is weaker / needs to improve

- GALILEO should explicitly measure and report the **trace–output gap** (when traces are available) as a distinct failure mode from “wrong reasoning”.
- Consider capability-dependent effects like **inverse scaling** when selecting model tiers and stress levels.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a “final output gap” diagnostic: when a model explains/derives A but outputs B (especially B matching social pressure), count it separately.
- [ ] Add a “discrimination” split: adversarial hints vs *helpful/correct* hints; report acceptance of helpful hints.
- [ ] Add an “inverse scaling” plot: sycophancy vs model tier *stratified by task difficulty*.
- [ ] In related work, position GALILEO vs **process verification** approaches (trace-based verification / external judging).

## Quotes / details to potentially cite

- “Sycophancy manifests as trace-output inconsistency: models derive one answer but output another to please users.”
- Reported headline numbers: “0.0% sycophancy” and “88% acceptance of valid hints” (CAP discrimination test).
- Named failure modes: **Inverse Scaling** and **Final Output Gap**.
