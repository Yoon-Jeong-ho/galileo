# ICON: Intent-Context Coupling for Efficient Multi-Turn Jailbreak Attack

- Year: 2026
- Venue: arXiv
- Authors: Xingwei Lin; Wenhao Lin; Sicong Cao; Jiahao Yu; Renke Huang; Lei Xue; Chunming Wu
- URL: https://arxiv.org/abs/2601.20903
- BibTeX key (if we add it): icon_intent_context_coupling_2026
- Tags: multi-turn, jailbreak, context-routing, drift, robustness, safety

## One-sentence takeaway

ICON argues that LLM safety is *context-dependent* ("intent–context coupling") and uses intent-driven routing into an authoritative context plus hierarchical optimization to achieve very high multi-turn jailbreak ASR.

## What problem does it solve?

- Efficiently constructing *effective* multi-turn jailbreak conversations (vs. slow step-by-step incremental probing) and avoiding attacks getting stuck in semantically mismatched contexts.

## What is the core method / protocol?

- Empirical observation/hypothesis: **Intent–Context Coupling** — when a malicious intent is paired with a semantically congruent context pattern, the model’s safety constraints relax (helpfulness/coherence trade-off shifts).
- **ICON framework**:
  - **Intent routing**: classify/identify malicious intent (e.g., hacking) and route it to a congruent **context pattern** (example given: "Scientific Research").
  - **Authoritative-style instantiation**: generate a *prompt sequence* (multi-turn) using an authoritative template (example given: "Academic Paper") that progressively builds context and then elicits prohibited content.
  - **Hierarchical Optimization Strategy**:
    - *Local/tactical*: refine prompts when an attempt fails.
    - *Global/strategic*: switch the broader context pattern when local refinement fails due to semantic incompatibility.

## What are the key metrics?

- Attack Success Rate (ASR) for multi-turn jailbreak attacks (across multiple target LLMs).
- Also discusses efficiency/query cost qualitatively (motivated as reducing step-by-step interaction), but ASR is the headline.

## What are the main results?

- Reports **state-of-the-art average ASR of 97.1%** across **eight** representative SOTA LLMs (commercial + open).

## How is this similar to GALILEO?

- Highlights a **multi-turn phenomenon** where behavior changes across turns due to the evolving dialogue state/context.
- Emphasizes **context drift / trajectory effects**: failures can arise from being in the “wrong” conversational region, not just wording.
- Useful as a cautionary example that single-turn robustness (or alignment) may not predict multi-turn robustness.

## How is this different from GALILEO?

- ICON is an **attack/red-teaming** paper targeting safety policy circumvention; GALILEO is about *robust multi-turn behavior under pressure* (e.g., sycophancy/persuasion, belief revision vs drift controls).
- ICON optimizes for **eliciting prohibited content**; GALILEO likely optimizes/assesses **stability and correctness** under repeated interaction.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO’s evaluation is framed around *truthfulness / belief stability* with clearly defined “good flips vs bad flips,” it may provide cleaner behavioral diagnostics than ASR-only jailbreak metrics.

## Where GALILEO is weaker / needs to improve

- Might under-emphasize **context-pattern sensitivity**: even if a model is stable in neutral contexts, authoritative-role or domain-framed contexts may induce different pressure dynamics.
- Needs to ensure tests cover **trajectory-level failures** (getting stuck in a bad conversational basin) rather than only per-turn responses.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add (or explicitly discuss) **context-pattern stress tests**: same user pressure goal, but framed inside different “authoritative” contexts (academic paper, clinical, legal, scientific research) to see if robustness claims hold.
- [ ] In GALILEO writeup, include a short related-work note: safety/robustness can be **asymmetric and context-dependent**, motivating multi-context evaluation.
- [ ] Consider a GALILEO metric slice: *robustness conditional on context family* (context-stratified flip/drift rates).

## Quotes / details to potentially cite

- “We characterize the Intent-Context Coupling phenomenon, revealing that LLM safety constraints are significantly relaxed when a malicious intent is coupled with a semantically congruent context pattern.” (Abstract)
- “ICON … achieving a state-of-the-art average Attack Success Rate (ASR) of 97.1%.” (Abstract)
