# Authenticated Workflows: A Systems Approach to Protecting Agentic AI

- Year: 2026
- Venue: arXiv
- Authors: Mohan Rajagopalan et al. (see arXiv for full author list)
- URL: https://arxiv.org/abs/2602.10465
- BibTeX key (if we add it): rajagopalan2026authenticatedworkflows
- Tags: agents, tool-use, workflow-security, authorization, cryptographic-attestation, policy-language

## One-sentence takeaway

A protocol-layer “trust layer” for agentic AI workflows that enforces policy intent + cryptographic integrity at every boundary crossing (prompts/tools/data/context), aiming for deterministic security (accept only operations with valid proofs).

## What problem does it solve?

- Enterprise agentic AI defenses (guardrails, semantic filters) are probabilistic, bypassable, and don’t compose across multi-turn workflows.
- Agents cross multiple heterogeneous components (LLM interface + orchestrator + tools + data sources) where attacks can chain across surfaces (prompt injection → tool misuse → data exfiltration → context poisoning).

## What is the core method / protocol?

- Models an agentic system as a distributed system with **four fundamental boundaries**: **prompts**, **tools**, **data**, **context**.
- Places distributed **Policy Enforcement Points (PEPs)** at each boundary crossing.
- Enforces two properties at crossings:
  - **Intent**: operation satisfies organizational policies.
  - **Integrity**: operation is authentic / unmodified via **cryptographic proofs/attestations**.
- Introduces **MAPL**, an “AI-native” policy language supporting:
  - compositional/inherited policies (intersection-style composition),
  - hierarchical scaling claims (O(log M + N) vs O(M×N)),
  - **attestation-gated workflows** (e.g., “export only if anonymization attestation exists”).
- Claims universal deployability via thin adapters (200–500 LOC) across multiple agent frameworks without protocol modifications.

## What are the key metrics?

- Attack/violation detection quality in their test suite: recall/false positives.
- Coverage against OWASP Top 10 categories (agentic-security mapping).
- Overhead (they claim sub-millisecond verification at enforcement points).

## What are the main results?

- Reports **100% recall** with **0 false positives** across **174 test cases**.
- Claims protection against **9/10 OWASP Top 10 risks**.
- Claims full mitigation of **two production CVEs** (named in the paper as OpenAI Atlas and GitHub MCP-related issues).
- Formal claims: completeness/minimality of the 4-boundary model + soundness of distributed enforcement + properties of policy composition.

## How is this similar to GALILEO?

- Shares the general framing that **multi-turn agentic systems fail at boundary management** (context accumulation, tool invocation risk, data-as-instructions issues).
- Useful as adjacent related work if GALILEO discusses **robustness/safety** of multi-step agent behavior and wants a “systems/security” neighbor.

## How is this different from GALILEO?

- Focuses on **cryptographic authenticity + authorization/policy enforcement** rather than behavioral robustness metrics, persuasion/sycophancy, or learning-based defenses.
- Assumes you can restructure the runtime to require proofs/attestations, rather than only observing model behavior.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO is primarily an *evaluation/behavioral* paper: GALILEO can apply to deployed black-box models without requiring new cryptographic infrastructure.

## Where GALILEO is weaker / needs to improve

- If GALILEO argues for safety in enterprise settings, this paper highlights that **probabilistic filters are structurally insufficient** for certain threat models; reviewer may ask why GALILEO doesn’t include “deterministic” controls (authn/authz, provenance) as baselines/assumptions.

## Action items for GALILEO (experiments / method / writing)

- [ ] Related-work paragraph: position “authenticated workflows / cryptographic trust layer” as complementary to behavioral robustness—i.e., GALILEO measures/mitigates behavioral failure modes, while authenticated workflows constrain what actions can execute.
- [ ] If GALILEO includes tool-use: add a short discussion of **policy enforcement points** and why behavioral metrics alone don’t prevent unauthorized actions.

## Quotes / details to potentially cite

- “Security reduces to protecting four fundamental boundaries: prompts, tools, data, and context.” (abstract)
- “Deterministic security—operations either carry valid cryptographic proof or are rejected.” (abstract)
- MAPL scaling claim: “O(log M + N) policies versus O(M×N) rules through hierarchical composition with cryptographic attestations…” (abstract)
