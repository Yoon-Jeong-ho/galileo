# Agent Skills for Large Language Models: Architecture, Acquisition, Security, and the Path Forward

- Year: 2026
- Venue: arXiv (survey)
- Authors: Renjun Xu; Yang Yan
- URL: https://arxiv.org/abs/2602.12430
- BibTeX key (if we add it): Xu2026AgentSkills
- Tags: agents, skills, mcp, progressive-disclosure, security, survey

## One-sentence takeaway

A focused survey that treats “agent skills” as a distinct abstraction layer (procedural packages + progressive disclosure) and argues that security/governance for community skill ecosystems is already a first-order issue.

## What problem does it solve?

- Clarifies the emerging “skill” layer for LLM agents (beyond prompts, RAG, and tool/function calling) and organizes recent work into a taxonomy.
- Surfaces security and lifecycle governance concerns for sharing/executing third-party skills at scale.

## What is the core method / protocol?

- Survey / taxonomy along four axes:
  - Architecture: filesystem-packaged skills (SKILL.md), progressive context loading (“progressive disclosure”), and relationship to Model Context Protocol (MCP).
  - Acquisition: RL with skill libraries, autonomous skill discovery, compositional synthesis.
  - Deployment: computer-use agent stacks, GUI grounding, benchmarks like OSWorld and SWE-bench.
  - Security: empirical vulnerability findings in community skills; proposes a “Skill Trust and Lifecycle Governance Framework” with gate-based permission tiers.

## What are the key metrics?

- Not a single new benchmark; highlights metrics used in adjacent areas (OSWorld / SWE-bench, etc.).
- Security highlight: cites an empirical finding that **26.1% of community-contributed skills contain vulnerabilities** (paper’s summary claim).

## What are the main results?

- Positions skills as “procedural knowledge packages” (instructions + code + resources) that can be loaded on demand, distinct from tools (execution) and RAG (passive text retrieval).
- Argues the stack is converging around (Skills + MCP) as complementary layers: skills provide “what to do”, MCP provides “how to connect”.
- Pushes a security/governance framing: provenance → verification gates → permissioned deployment capabilities.

## How is this similar to GALILEO?

- Overlaps strongly if GALILEO’s core contribution is a *skill packaging / progressive loading / tool-connection* story.
- The “skills vs MCP” separation is a useful lens for explaining architecture cleanly (workflow/procedure vs connectivity).

## How is this different from GALILEO?

- This is a survey (and a governance proposal), not a concrete system with new empirical wins.
- Emphasis is broad ecosystem mapping; likely minimal implementation detail beyond describing common patterns.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO includes a reproducible implementation + evaluations, it can claim concrete evidence where this survey is primarily definitional/taxonomic.
- GALILEO can potentially offer a more formal/operational definition of skill interfaces and loading policies.

## Where GALILEO is weaker / needs to improve

- Security posture: this survey makes it hard to ignore skill supply-chain risk; GALILEO should preempt reviewer concerns with explicit threat model + mitigations.
- Governance: need clear stance on permissions, provenance, and execution isolation for third-party skills.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a short “skills vs MCP (or tool connectors)” architectural diagram and narrative: procedural package vs connectivity layer.
- [ ] Add a threat model section for skills (prompt injection inside skills, malicious scripts, data exfiltration via connectors), plus mitigations (sandboxing, allowlists, capability-based permissions).
- [ ] If we support community skills: propose a lightweight provenance/tier system (similar to their gate-based permission tiers) and document what each tier can do.
- [ ] Consider adding a “progressive disclosure” ablation: eager-load-all vs staged loading, measuring context cost and performance.

## Quotes / details to potentially cite

- “Agent skills—composable packages of instructions, code, and resources that agents load on demand—enable dynamic capability extension without retraining.” (Abstract)
- The survey summary claim that **26.1% of community-contributed skills contain vulnerabilities** (useful motivation for governance/permissions).