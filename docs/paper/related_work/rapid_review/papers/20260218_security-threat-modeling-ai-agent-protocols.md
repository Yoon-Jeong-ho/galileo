# Security Threat Modeling for Emerging AI-Agent Protocols: A Comparative Analysis of MCP, A2A, Agora, and ANP

- Year: 2026
- Venue: arXiv
- Authors: Zeynab Anbiaee (per arXiv)
- URL: https://arxiv.org/abs/2602.11327
- BibTeX key (if we add it): anb iaee2026agent-protocol-threat-modeling
- Tags: agents, protocols, security, threat-modeling, mcp, a2a, anp, agora

## One-sentence takeaway

A protocol-centric threat-modeling + qualitative risk-assessment framework for emerging agent communication protocols (MCP/A2A/Agora/ANP), plus an MCP case study quantifying “wrong-provider tool execution” risk under multi-server composition.

## What problem does it solve?

- Agent ecosystems increasingly rely on standardized communication protocols (tool calling, inter-agent messaging, cross-org interoperability), but protocol-level security principles and threat modeling are fragmented and non-standardized.
- There is no shared, protocol-centric framework to compare risks across protocols and across lifecycle phases (creation/operation/update).

## What is the core method / protocol?

- Structured threat modeling that explicitly examines:
  - protocol architecture and trust assumptions
  - interaction patterns and lifecycle behaviors
  - trust boundaries and identity/authorization binding assumptions
  - cross-protocol / multi-server composition risk surfaces
- A qualitative risk assessment framework:
  - identifies 12 protocol-level risks (paper-defined)
  - scores likelihood/impact and overall risk
  - evaluates posture across creation, operation, and update phases
- Measurement-driven MCP case study:
  - formalizes lack of mandatory validation/attestation for executable components as a falsifiable claim
  - measures wrong-provider tool execution under multi-server composition across “representative resolver policies”

## What are the key metrics?

- Qualitative: likelihood × impact style scoring across lifecycle phases (creation/operation/update).
- For MCP case study: rate / frequency of wrong-provider tool execution under multi-server composition as a function of resolver policy (measurement-driven).

## What are the main results?

- Proposes a comparative taxonomy of protocol-induced risk surfaces across MCP, A2A, Agora, ANP.
- Produces a lifecycle-aware qualitative assessment of protocol risks (12 risks, per paper).
- Demonstrates (via MCP case study) that missing mandatory validation/attestation can be operationalized and measured as wrong-provider tool execution in multi-server settings, and that resolver policy affects this risk.

## How is this similar to GALILEO?

- Overlaps with GALILEO’s concerns about safe tool invocation and the security of agent→tool/agent→agent interaction surfaces.
- Highlights composition risks (multi-tool / multi-provider) that also matter for GALILEO-style “agent + tool ecosystem” deployments.

## How is this different from GALILEO?

- This is protocol-security / threat-modeling work rather than proposing a new agent method or optimization objective.
- Focus is comparative analysis across protocols and qualitative risk frameworks, not model training/evaluation for task performance.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO provides explicit, end-to-end safety invariants and evaluation on task correctness + safety under adversarial tool environments, it may offer more concrete (quantitative) guarantees than qualitative risk scoring alone.

## Where GALILEO is weaker / needs to improve

- GALILEO writing should explicitly acknowledge protocol-level threat surfaces (identity binding, update channel, tool provenance/attestation, multi-provider composition) and articulate assumptions/mitigations.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a “protocol & ecosystem assumptions” subsection: identity/auth binding, tool provenance, server composition model, update channel trust.
- [ ] Consider an experiment that mimics the MCP “wrong-provider tool execution” phenomenon: multi-provider tool catalogs + resolver policies; measure misbinding / wrong-tool execution and downstream harm.
- [ ] In related work, cite this paper as evidence that protocol-level risk assessment for agent protocols is still immature and that missing validation/attestation is an actionable risk.

## Quotes / details to potentially cite

- “no protocol-centric risk assessment framework has been established yet” (abstract claim)
- Contributions summary (intro): structured comparative threat modeling; architecture-based catalog of threat hypotheses; lifecycle-aware qualitative risk assessment; measurement-driven MCP case study quantifying wrong-provider tool execution under multi-server composition.
