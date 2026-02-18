# OMNI-LEAK: Orchestrator Multi-Agent Network Induced Data Leakage

- Year: 2026
- Venue: arXiv (preprint; under review ICML 2026)
- Authors: Jay J Culligan; Yarin Gal; Philip Torr; Rahaf Aljundi; Alasdair Paren; Adel Bibi
- URL: https://arxiv.org/abs/2602.13477
- BibTeX key (if we add it): Culligan2026OmniLeak
- Tags: multi-agent, orchestrator, prompt-injection, data-leakage, access-control, red-teaming

## One-sentence takeaway

A single *indirect prompt injection* into a public data source can cascade through an orchestrator-style multi-agent system to exfiltrate private data (e.g., SSNs) *despite access control*, and most tested frontier models are vulnerable in at least one configuration.

## What problem does it solve?

- Threat modeling + empirical security evaluation for **orchestrator/worker multi-agent** patterns, focusing on **privacy/data leakage** (rather than only code execution).
- Shows that **single-agent safety conclusions don’t reliably generalize** to multi-agent setups with delegation + tool boundaries.

## What is the core method / protocol?

- A concrete orchestrator setup with:
  - **Orchestrator agent** that routes tasks.
  - **SQL agent** that answers benign HR-style questions by querying a DB.
  - **Notification agent** that can send emails.
- Data is split into **public** and **private** tables; access control prevents unprivileged users from even seeing the private table.
- **OMNI-LEAK attack** (multi-step cascade): attacker injects malicious instructions into a **public text field** (e.g., department_name). When a *privileged* user later queries data touching that field, the SQL agent ingests the injection, retrieves private data, then persuades the orchestrator to exfiltrate via the notification agent.
- Benchmarking protocol:
  - 10 indirect-injection attacks grouped into 4 “persuasion tactic” categories.
  - Each attack has an **explicit** version (attacker uses schema knowledge) and an **implicit** version (no schema knowledge).
  - 3 DB sizes (Toy/Medium/Big).
  - 5 benign user queries; 10 repeats each; temperature 1.
  - Automated success check by **exact string match** on expected leaked data.

## What are the key metrics?

- **BA**: benign query accuracy with no attacks.
- **RA**: benign query accuracy with an attack present (robust benign query accuracy).
- **E**: expected number of queries needed for a successful attack (lower = easier attack), computed from success rate; ∞ indicates no successes.

## What are the main results?

- **All models except Claude Sonnet 4** were vulnerable to at least one OMNI-LEAK attack (in their tested settings).
- Explicit vs implicit: explicit typically easier, but **implicit attacks can still work** (schema hiding provides limited protection).
  - Example cited: gemini-2.5-flash had E ≈ 17/17/9 (explicit; toy/med/big) vs 18/20/14 (implicit).
- **Database size** had **little effect on attack success** (E not consistently increasing with size), implying large production DBs are still at risk.
  - Example cited: gpt-4.1-mini E ≈ 6/4/6 (explicit) and 7/7/9 (implicit) across toy/med/big.
- Under attack, **RA tends to drop with DB size** (attacks interfere more with normal behavior), potentially making attacks more noticeable.
  - Example cited: o4-mini RA dropped from ~90.4% → 84.0% → 76.4% as DB size increased (implicit setting table).
- System-level finding: **downstream agent choice can dominate overall vulnerability**. A weak SQL agent can paraphrase/“launder” the malicious instruction so the orchestrator can’t detect it, even if the orchestrator model is comparatively robust.

## How is this similar to GALILEO?

- Both are about **multi-step, multi-turn robustness failures** under adversarially inserted content.
- Highlights that **intermediate representations/messages** (agent-to-agent outputs) can “rewrite” or conceal the original adversarial intent—relevant to any evaluation of downstream drift and auditing signals.

## How is this different from GALILEO?

- Focus is **security/data exfiltration** in tool-using orchestrator systems (SQL + email), not conversational belief change / persuasion / stance drift.
- Success metric is a **binary leakage event** (string-match) rather than trajectory-level belief stability, recovery, or calibration.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO is centered on *behavioral drift under pressure* (rather than tool exfiltration), we can offer cleaner controls for:
  - pressure-only vs evidence-driven updates,
  - recovery dynamics,
  - richer turn-by-turn trajectory metrics.

## Where GALILEO is weaker / needs to improve

- If GALILEO involves agentic/tool settings, this paper suggests we need to explicitly cover:
  - **cross-agent “instruction laundering”** (malicious intent paraphrased into innocuous-looking messages),
  - evaluation that conditions on **privileged vs unprivileged** contexts,
  - outcomes that are *indirect* (exfil via another tool) rather than direct model output.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a threat-model paragraph: why **multi-agent composition breaks single-agent safety assumptions** (cite OMNI-LEAK as empirical evidence).
- [ ] If GALILEO uses multi-agent pipelines, add an ablation: hold orchestrator fixed, vary downstream agent, and measure how often downstream outputs “launder” adversarial intent.
- [ ] Consider a “privilege boundary” evaluation variant (privileged user triggers attack vs attacker directly prompting), to mirror the access-control bypass setting.

## Quotes / details to potentially cite

- OMNI-LEAK definition (high-level): a single indirect prompt injection can compromise SQL agent → orchestrator → notification agent to leak sensitive data **even with data access control**.
- “All models, except claude-sonnet-4, are vulnerable to at least one OMNI-LEAK attack.”
- “Database size does not appear to impact attack success.”
- Risk framing from conclusion: low success rates can still be operationally catastrophic (e.g., 1/500 in a 100-person company can leak within days).
