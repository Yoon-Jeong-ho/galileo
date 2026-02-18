# TAMAS: Benchmarking Adversarial Risks in Multi-Agent LLM Systems

- Year: 2025
- Venue: ICML 2025 MAS Workshop (arXiv preprint)
- Authors: Ishan Kavathekar; Hemang Jain; Ameya Rathod; Ponnurangam Kumaraguru; Tanuja Ganu
- URL: https://arxiv.org/abs/2511.05269
- BibTeX key (if we add it): tamas2025
- Tags: multi-agent, adversarial-risk, benchmark, prompt-injection

## One-sentence takeaway

TAMAS is an early multi-agent-specific safety benchmark that operationalizes six adversarial attack types (incl. collusion/Byzantine/contradiction) across domains and frameworks, and proposes an “Effective Robustness Score” to quantify the safety–utility tradeoff.

## What problem does it solve?

- Existing agent-safety benchmarks mostly target **single-agent** settings and/or narrow threat types (direct/indirect prompt injection, risky code exec), missing **emergent vulnerabilities** from multi-agent coordination.
- Provides a structured way to stress-test multi-agent LLM systems under adversarial conditions across domains and agent architectures.

## What is the core method / protocol?

- Build a benchmark, **TAMAS**, with:
  - 5 scenarios / domains: education, legal, finance, healthcare, news.
  - 300 adversarial instances + 100 harmless tasks.
  - 6 attack types spanning prompt-/environment-/agent-level surfaces:
    - Impersonation
    - Direct prompt injection (DPI)
    - Indirect prompt injection (IPI)
    - Contradicting agents
    - Byzantine agent
    - Colluding agents
  - 211 tools referenced/used in tasks.
- Evaluate multiple multi-agent frameworks/configurations:
  - Frameworks: AutoGen, CrewAI.
  - 3 interaction configurations (centralized vs decentralized variants; paper compares configuration robustness).
  - 10 backbone LLMs.
- Introduce **Effective Robustness Score (ERS)** to quantify tradeoff between:
  - task effectiveness (utility / success on benign goal)
  - and safety/robustness under attack (resistance to malicious goal following).

## What are the key metrics?

- Attack success / system failure rates under each attack type (as reported in their experiments).
- Task success on harmless tasks.
- ERS: a combined score intended to summarize the safety–effectiveness tradeoff (details/weighting to check if we later cite precisely).

## What are the main results?

- Multi-agent systems are **highly vulnerable** across the tested adversarial vectors; robustness varies substantially by:
  - backbone LLM choice
  - agent framework (AutoGen vs CrewAI)
  - interaction configuration.
- Multi-agent-specific attacks (collusion / contradiction / compromised-agent behaviors) expose failure modes not captured by single-agent benchmarks.

## How is this similar to GALILEO?

- Both are concerned with **robustness/safety evaluation** of agentic LLM systems under adversarial pressure.
- Highlights that architecture/protocol choices can systematically change failure rates (a framing useful for GALILEO evaluation sections).

## How is this different from GALILEO?

- TAMAS is primarily a **benchmark/dataset + measurement** paper (task suite, attack taxonomy, empirical comparison), not a new defense or core inference method.
- Focus is specifically on **multi-agent** vulnerabilities (incl. compromised-agent behaviors), while GALILEO’s core contribution may not require multi-agent assumptions.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO provides clearer causal controls (e.g., separating evidence-driven updates vs drift) or stronger protocols/metrics (e.g., time-to-failure / recovery), that would complement TAMAS’s broader-but-coarser benchmark framing.

## Where GALILEO is weaker / needs to improve

- If GALILEO claims broad agent-safety relevance, it may need to explicitly discuss (or test) **multi-agent attack surfaces** (collusion/Byzantine/contradiction), which TAMAS surfaces as practically important.

## Action items for GALILEO (experiments / method / writing)

- [ ] In related work, cite TAMAS as a multi-agent safety benchmark and emphasize the gap between single-agent vs multi-agent vulnerabilities.
- [ ] Consider adding a short discussion subsection: “Multi-agent extensions / threats” (impersonation, collusion, contradiction) and whether GALILEO’s evaluation protocol would generalize.
- [ ] If feasible, add a small experiment: run GALILEO-style evaluation on at least one multi-agent framework/config (or motivate why out-of-scope).

## Quotes / details to potentially cite

- “TAMAS includes five distinct scenarios comprising 300 adversarial instances across six attack types and 211 tools, along with 100 harmless tasks.” (arXiv abstract)
- “We introduce Effective Robustness Score (ERS) to assess the tradeoff between safety and task effectiveness…” (arXiv abstract)
