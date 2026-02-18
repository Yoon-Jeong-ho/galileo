# Cognitive Control Architecture (CCA): A Lifecycle Supervision Framework for Robustly Aligned AI Agents

- Year: 2025
- Venue: arXiv
- Authors: Zhibo Liang; Tianze Hu; Zaiye Chen; Mingjie Tang
- URL: https://arxiv.org/abs/2512.06716
- BibTeX key (if we add it): Liang2025CCA
- Tags: agents, indirect-prompt-injection, supervision, robustness, control-flow-integrity, data-flow-integrity

## One-sentence takeaway

CCA proposes end-to-end ("full lifecycle") supervision for LLM agents against indirect prompt injection by (1) enforcing an expected plan via an Intent Graph and (2) escalating suspicious deviations to a tiered LLM adjudicator that scores intent alignment.

## What problem does it solve?

- Indirect Prompt Injection (IPI) against tool-using agents, where malicious instructions are embedded in external data sources and cause unauthorized tool invocations and goal hijacking.
- Argues many defenses are fragmented (point checks at runtime, isolation, or training-time hardening) and therefore force a security/functionality/efficiency trade-off without end-to-end integrity.

## What is the core method / protocol?

- Key premise: even subtle IPI ultimately manifests as a *detectable deviation in the agent’s action trajectory* relative to the legitimate plan.
- **Intent Graph (proactive supervision):** pre-generates an expected cognitive / tool-use flow (control-flow + data-flow integrity). Used as a deterministic validator to catch overt deviations efficiently.
- **Tiered Adjudicator (reactive deep check):** when a deviation is detected, invoke an LLM-based adjudicator that performs deeper reasoning.
  - Uses an **Intent Alignment Score** that combines multiple signals (paper describes semantic similarity, causal contribution, and source trustworthiness) to decide whether the deviation is benign/necessary vs malicious.
- Architecture is explicitly “defense in depth”: Layer 1 is code-based validation; Layer 2 is semantic adjudication.

## What are the key metrics?

- Attack success / defense success on agent security benchmarks (evaluated on **AgentDojo** in the paper).
- Functional utility under defense (ability to complete benign tasks while defended).
- Efficiency / overhead (API calls / runtime overhead), with the tiered scheme intended to reduce heavy adjudication calls.

## What are the main results?

- On AgentDojo, CCA reportedly withstands sophisticated IPI attacks that defeat other advanced defenses, while maintaining efficiency (reduced deep-check overhead by only escalating on deviations).
- Claims to reconcile the security–functionality–efficiency trade-off better than prior fragmented approaches.

## How is this similar to GALILEO?

- If GALILEO is targeting robust tool-using agents / instruction-data separation / IPI-style threats, CCA is in the same design space: *trajectory-level* monitoring rather than only single-step filtering.
- The Intent Graph resembles a plan-/dependency-graph style safety envelope, and the adjudicator resembles an escalation-based “expensive check only when needed” pattern.

## How is this different from GALILEO?

- CCA’s core framing is *full-lifecycle cognitive supervision* with explicit **control-flow + data-flow integrity** and a deviation-triggered adjudication mechanism.
- Emphasis on a bespoke multi-signal **Intent Alignment Score** (semantic + causal + trust signals) for adjudicating deviations, rather than purely rule-based tool allowlists or purely learned detectors.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO provides a more formal threat model, clearer invariants, or stronger guarantees than heuristic scoring, that could be a stronger story than CCA’s primarily architectural proposal.
- If GALILEO has simpler primitives than “intent graph generation + scoring-based adjudication,” it may be easier to implement and reason about.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently lacks explicit data-flow integrity tracking (taint/trust provenance) or lacks a principled escalation layer for ambiguous deviations, CCA highlights a concrete gap.

## Action items for GALILEO (experiments / method / writing)

- [ ] In related work: position against “plan/dependency-graph” defenses (control-flow integrity) and emphasize why adding provenance/trust signals or escalation reasoning matters for conditional attacks.
- [ ] Consider adding a “tiered adjudication” ablation: cheap deterministic guard most of the time; expensive semantic adjudication only on deviations.
- [ ] Consider incorporating a source trustworthiness / provenance feature into any intent-alignment or deviation scoring.

## Quotes / details to potentially cite

- Abstract-level framing: “full-lifecycle cognitive supervision” via (i) pre-generated Intent Graph for control-/data-flow integrity and (ii) a Tiered Adjudicator using multi-dimensional scoring to counter conditional attacks.
- Benchmark note: experiments are reported on AgentDojo.
