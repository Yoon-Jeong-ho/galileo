# LLM Output Drift: Cross-Provider Validation & Mitigation for Financial Workflows

- Year: 2025
- Venue: AI4F @ ACM ICAIF ’25 (workshop) / arXiv
- Authors: Raffi Khatchadourian (others not listed in the arXiv HTML extract)
- URL: https://arxiv.org/abs/2511.07585
- BibTeX key (if we add it): Khatchadourian2025OutputDriftFinance (suggested)
- Tags: drift, nondeterminism, reproducibility, compliance, RAG, evaluation, monitoring

## One-sentence takeaway

In regulated financial tasks, the paper finds that some small/open models can be fully deterministic under greedy decoding while a much larger model remains highly inconsistent, and proposes an audit-style harness + invariants + dual-provider validation to manage “output drift.”

## What problem does it solve?

- Financial workflows need auditability and repeatability, but LLM outputs can vary across repeated runs (“output drift”), undermining compliance and trust.
- The paper frames this as a deployment/verification problem: you need a test harness and acceptance criteria (invariants) that are compatible with regulated tasks.

## What is the core method / protocol?

- Deterministic harness: greedy decoding (temperature 0), fixed seeds, and a deterministic retrieval ordering (SEC 10‑K structure-aware) for RAG.
- Invariant checkers tailored to output types:
  - RAG: validate SEC citations and tolerate small numeric differences via a “materiality” threshold (±5%).
  - JSON / SQL: validate exact/structural correctness (structured outputs are expected to be more stable).
- “Three-tier” model classification for deployment decisions (Tier 1 fully compliant; Tier 2 limited; Tier 3 requires additional validation).
- Cross-provider validation / attestation: confirm deterministic behavior holds across local vs cloud deployments, and use dual-provider checks for risk mitigation.

## What are the key metrics?

- Output consistency rate across repeated trials (paper mentions n=16 per condition; 480 total runs).
- Confidence intervals for consistency and significance tests (Fisher’s exact test reported).
- Task-type sensitivity: structured (SQL) vs unstructured (RAG) consistency.

## What are the main results?

- Strong scale-related claim (as reported): smaller models (e.g., Granite‑3‑8B, Qwen2.5‑7B) reached 100% consistency at T=0; a 120B model (“GPT‑OSS‑120B”) showed very low consistency (~12.5%) even with identical configuration.
- Structured tasks (SQL) were comparatively stable even at higher temperature (T=0.2), while RAG tasks exhibited substantial drift (reported 25–75% range).
- Deterministic behavior can “transfer” across provider contexts when the harness and constraints are enforced.

## How is this similar to GALILEO?

- Shares the core concern: stability/robustness across multi-run / multi-turn (here: repeated runs and provider contexts rather than interactive turns).
- Emphasizes evaluation protocols and metrics (consistency, invariants) rather than only model capability.
- Highlights that task structure matters for robustness (structured vs unstructured), which parallels GALILEO’s “when do models break” framing.

## How is this different from GALILEO?

- Focuses on *run-to-run nondeterminism and compliance* in financial workflows, not conversational multi-turn drift under adversarial or social pressure.
- Uses domain-specific acceptance criteria (SEC citation validation, ±5% materiality) rather than general-purpose multi-turn robustness metrics (e.g., survival/time-to-failure).
- The “tiering” is a deployment decision framework, not a behavioral robustness taxonomy.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO’s metrics are model/task-family general (not domain-specific), it may generalize better beyond finance.
- GALILEO can likely capture *trajectory-level* failures (turn-of-failure, recovery) that this paper does not target.

## Where GALILEO is weaker / needs to improve

- This paper provides a concrete compliance-style harness + invariants + attestation framing; GALILEO could benefit from a similarly “auditable” protocol packaging.
- GALILEO may need clearer guidance on deterministic settings + invariants for structured outputs (JSON/SQL) vs free-form outputs.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a short “auditability / determinism” subsection in related work: distinguish *multi-turn behavioral drift* vs *run-to-run output nondeterminism*.
- [ ] Consider adding a small protocol note: for structured tasks, evaluate consistency under T=0.2 too (to show robustness margins), mirroring their SQL stability observation.
- [ ] If we discuss mitigation, mention invariant checking (schema checks, citation validation, tolerances) as a practical complement to behavioral metrics.

## Quotes / details to potentially cite

- “We quantify drift across five model architectures (7B–120B parameters) on regulated financial tasks…” (Abstract)
- “…smaller models … achieve 100% output consistency at T=0.0, while [a 120B model] exhibits only 12.5% consistency…” (Abstract)
- “…structured tasks (SQL) remain stable even at T=0.2, while RAG tasks show drift (25–75%)…” (Abstract)
