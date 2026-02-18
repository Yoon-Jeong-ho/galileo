# Agent Tools Orchestration Leaks More: Dataset, Benchmark, and Mitigation

- Year: 2025
- Venue: arXiv
- Authors: Yuxuan Qiao; Dongqin Liu; Hongchang Yang; Wei Zhou; Songlin Hu
- URL: https://arxiv.org/abs/2512.16310
- BibTeX key (if we add it): qiao2025topr
- Tags: agents, tool-use, privacy, leakage, benchmark, mitigation

## One-sentence takeaway

Single-agent multi-tool LLM agents can *autonomously* combine innocuous tool outputs into sensitive inferences (TOP-R); TOP-Bench + H-Score quantify the safety/robustness trade-off, and a principle-style prompt mitigation (PEP) reduces leakage substantially.

## What problem does it solve?

- Identifies and formalizes an *interaction-level* privacy failure mode for tool-using agents: sensitive attributes can be inferred via multi-step orchestration across tools even when each individual tool query/output seems benign (a “mosaic effect” in an agentic setting).
- Provides an evaluation benchmark (TOP-Bench) and metrics to measure (i) privacy leakage prevalence and (ii) whether mitigations preserve task robustness.

## What is the core method / protocol?

- **Formalization of TOP-R (Tools Orchestration Privacy Risk):** frames the root cause as a misaligned objective (helpfulness reward dominates privacy cost), plus autonomous trajectory generation and indiscriminate multi-source synthesis.
- **TOP-Bench:** paired *benign* vs *leakage* scenarios intended to test whether an agent will overreach and infer sensitive info while completing a benign user goal.
- **Counterfactual Cue:** introduces a causality-inspired “stress test” by injecting a benign alternative explanation that should logically negate the privacy inference; used to probe whether the agent respects the boundary or still “pattern-matches” to a sensitive conclusion.
- **Metrics:**
  - **Risk Leakage Rate (RLR):** fraction of cases where the agent produces the sensitive inference/leak.
  - **H-Score:** a holistic score intended to capture the safety–robustness trade-off (high when leakage is low without breaking benign-task competence).
- **Mitigation (PEP; Privacy Enhancement Principle):** a *principle-based prompt intervention* that tries to reshape the effective objective toward privacy-aware behavior.

## What are the key metrics?

- Risk Leakage Rate (RLR)
- H-Score (holistic safety/robustness)
- (Conceptual) Counterfactual Cue pass/fail behavior (whether injected counterfactual prevents the privacy inference)

## What are the main results?

- Across **8 representative models**, TOP-R appears severe: **average RLR = 90.24%** and **average H-Score = 0.167** (no model > 0.3).
- PEP mitigation improves privacy alignment substantially: **RLR reduced to 46.58%** and **H-Score increased to 0.624**.
- The paper argues an “**Intelligence–Privacy Paradox**”: stronger reasoning/competence can *increase* competence-driven privacy leakage when explicit privacy alignment is missing.

## How is this similar to GALILEO?

- Both focus on *multi-turn* / trajectory-level failure modes that arise from the agent’s optimization for “being helpful” under pressure/interaction, rather than from a single static model output.
- Both emphasize **paired controls** (their benign-vs-leakage pairing is analogous in spirit to GALILEO’s persona-pressure vs neutral re-asking control for separating effects).
- Both implicitly motivate that **capability ≠ alignment**: stronger models can fail “more dangerously” in context-sensitive ways.

## How is this different from GALILEO?

- Target failure mode differs:
  - This paper: **privacy inference/leakage** via *tool orchestration* (multi-tool, external observations).
  - GALILEO: **truth maintenance / flip dynamics** under *persona pressure* in ground-truth tasks (no tool-use requirement).
- Threat model differs: TOP-R is about synthesizing sensitive facts from fragments; GALILEO is about persuasion/pressure-induced drift without new evidence.
- Mitigation differs: PEP is a prompt-principle safety steering; GALILEO primarily aims to *measure* survival/TOF/recovery and compare conditions.

## Where GALILEO is stronger / cleaner (if true)

- GALILEO’s SSOT protocol centers on **ground-truth tasks** and yields clean, auditable metrics (survival/TOF/recovery) that directly connect to correctness.
- Neutral re-asking control in GALILEO explicitly isolates *drift without new evidence*, which is a complementary axis to TOP-R’s tool-evidence accumulation.

## Where GALILEO is weaker / needs to improve

- If GALILEO wants broader “agent safety” positioning, it may currently under-cover **tool-mediated, multi-source** interaction risks (privacy or overreach) that are increasingly central in deployed agents.

## Action items for GALILEO (experiments / method / writing)

- [ ] Related-work positioning: cite TOP-R as evidence that *trajectory-level* failures can worsen with capability; use as an analogy for “competence-driven” failure modes.
- [ ] Writing: consider a short paragraph contrasting **privacy leakage via evidence aggregation** (TOP-R) vs **truth drift under pressure without evidence** (GALILEO), reinforcing why GALILEO’s neutral re-asking control matters.
- [ ] (Optional) Future-work idea: extend GALILEO-style survival/TOF framing to other alignment axes (privacy boundaries, overreach) in tool-using settings.

## Quotes / details to potentially cite

- Definition framing (from abstract): “**Tools Orchestration Privacy Risk (TOP-R)**, where an agent … **aggregates information fragments across multiple tools** and … **synthesize[s] unexpected sensitive information**.”
- Key numbers (abstract): “average **Risk Leakage Rate (RLR)** … **90.24%** … average **H-Score** … **0.167** … PEP … reducing … to **46.58%** … improving … to **0.624**.”
