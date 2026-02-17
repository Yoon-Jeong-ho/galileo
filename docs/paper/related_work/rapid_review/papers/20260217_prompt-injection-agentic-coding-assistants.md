# Prompt Injection Attacks on Agentic Coding Assistants: A Systematic Analysis of Vulnerabilities in Skills, Tools, and Protocol Ecosystems

- Year: 2026
- Venue: arXiv (SoK / survey)
- Authors: Narek Maloyan; Dmitry Namiot
- URL: https://arxiv.org/abs/2601.17548
- BibTeX key (if we add it): Maloyan2026PromptInjectionAgenticCoding
- Tags: prompt-injection, agents, tool-use, MCP, security, survey

## One-sentence takeaway

A 2026 SoK that organizes prompt-injection attacks against tool-using/skill-based coding agents (incl. MCP ecosystems) into a 3D taxonomy and argues for architectural, defense-in-depth mitigations rather than filters.

## What problem does it solve?

- As agentic coding assistants gain file/shell/web/tool access, their attack surface expands beyond “chat prompt injection” into *tool/skill/protocol* compromise.
- The literature is fragmented across different attack/defense taxonomies; the paper aims to unify terminology, map the space, and summarize empirical findings.

## What is the core method / protocol?

- Systematization-of-knowledge + structured literature review (they report synthesizing 78 studies, 2021–2026).
- Proposes a three-dimensional taxonomy:
  - Delivery vectors (where malicious instructions live: e.g., repo/PR/issues/docs/web/pages/resources)
  - Attack modalities (e.g., indirect injection, tool poisoning, protocol exploitation, multimodal injection)
  - Propagation behaviors (how the attack spreads across context/tools/agents)
- Discusses MCP as a key “connector” layer (resources/prompts/tools) and highlights new protocol/ecosystem-specific vulnerabilities.

## What are the key metrics?

- Meta-analysis style summary metrics (as reported by prior work), e.g.:
  - Attack success rate vs. defenses (claims >85% success for adaptive attacks against SOTA defenses)
  - Defense mitigation rate / bypass rate (claims many defenses <50% against adaptive strategies)
- Note: these are *synthesized from cited benchmarks/studies* (they state they did not run replications).

## What are the main results?

- Consolidates a large attack catalog (abstract claims 42 techniques; intro section claims cataloging/extending prior taxonomies and adding MCP/protocol-level attacks).
- Argues most evaluated defenses are brittle to adaptive adversaries; recommends architectural defense-in-depth.
- Calls out “skill-based architecture” exploit chains (skills/extensions as a new analogue to browser-extension ecosystems).

## How is this similar to GALILEO?

- Shared theme: failures caused by *instruction-following over untrusted context*, especially in multi-step/tool-using settings.
- Provides vocabulary and threat-model framing for “context poisoning” and “cross-origin context” issues that can resemble drift/contamination in long-horizon agent runs.

## How is this different from GALILEO?

- Focuses on *security* (prompt injection, tool/protocol compromise), not truthfulness / belief revision / conversational drift metrics.
- Largely survey/taxonomy rather than proposing a specific evaluation protocol akin to GALILEO-style longitudinal robustness measurement.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO’s target is *measurement of stability/drift and recovery* in dialogue, it likely has clearer experimental controls (evidence vs. pressure vs. noise) than security-oriented SoKs.

## Where GALILEO is weaker / needs to improve

- If GALILEO touches agentic tool use, this paper suggests GALILEO should explicitly model:
  - Untrusted resource ingestion (docs/web/repo artifacts)
  - Skill/tool registry as an attack surface
  - Protocol-level threats (e.g., MCP-style tool schemas and server trust)

## Action items for GALILEO (experiments / method / writing)

- [ ] In related work, cite this as an SoK establishing prompt injection as an architectural vulnerability class for tool-using agents.
- [ ] Add a short paragraph contrasting “security prompt injection” vs. “conversational drift/belief revision” to clarify scope.
- [ ] If relevant, add an ablation/setting where the model reads *untrusted external content* mid-conversation and measure persistence/propagation of contamination.

## Quotes / details to potentially cite

- NIST characterization: prompt injection as “generative AI’s greatest security flaw” (as quoted/cited in their intro).
- OWASP LLM Top 10: prompt injection ranked #1 (as cited in their intro).
- Claims (from their meta-analysis) that adaptive attacks can exceed 85% success and many defenses mitigate <50% (use carefully; attribute as “reported by Maloyan & Namiot (2026), synthesizing prior studies”).
