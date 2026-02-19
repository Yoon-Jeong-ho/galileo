# Debate, Deliberate, Decide (D3): A Cost-Aware Adversarial Framework for Reliable and Interpretable LLM Evaluation

- Year: 2024 (arXiv); 2026 (EACL, per arXiv record)
- Venue: arXiv:2410.04663 (EACL 2026)
- Authors: Abir Harrasse et al. (see arXiv)
- URL: https://arxiv.org/abs/2410.04663
- BibTeX key (if we add it): harrasse2024d3
- Tags: llm-eval, llm-judge, multi-agent, debate, cost-aware, reliability, bias

## One-sentence takeaway
D3 turns LLM-judge evaluation into a structured, adversarial, multi-agent debate with explicit token budgets, improving agreement with humans and reducing common judging biases while trading off cost via early stopping.

## What problem does it solve?
- Automated LLM evaluation via model judges is often inconsistent and biased (position/verbosity effects), and lacks interpretability/transparent decision criteria.
- Practical evaluation must also consider *cost* (token usage) when increasing deliberation.

## What is the core method / protocol?
- A role-specialized multi-agent orchestration: multiple *advocates* argue for candidate answers; a *judge* decides; optionally a *jury* aggregates.
- Two complementary protocols:
  - **MORE (Multi-Advocate One-Round Evaluation):** elicit *k* parallel defenses/arguments for an answer; aggregate to amplify signal via diverse advocacy.
  - **SAMRE (Single-Advocate Multi-Round Evaluation) + budgeted stopping:** iterate advocate↔judge debate rounds, with explicit token budget, convergence checks, and early stopping.
- A probabilistic model over *score gaps* across rounds to reason about convergence/reliability and why parallel advocacy increases separation.
- Bias controls mentioned in abstract: anonymization + role diversification to reduce positional/verbosity biases.

## What are the key metrics?
- Agreement with human judgments on standard benchmarks (abstract mentions: accuracy and Cohen’s kappa).
- Bias measurements (positional bias, verbosity bias) and the effect of anonymization/role diversification.
- Cost–accuracy trade-off via token budgets and early stopping.

## What are the main results?
- On MT-Bench, AlignBench, and AUTO-J (per abstract), D3 reports state-of-the-art agreement with human judgments (accuracy / Cohen’s kappa).
- Reduced positional and verbosity biases relative to simpler judging setups (via anonymization + diversified roles).
- Favorable cost–accuracy frontier: iterative debate improves reliability, and budgeted stopping controls cost.

## How is this similar to GALILEO?
- Shares the theme of *reliable evaluation under constraints* and making evaluation *more interpretable* (structured reasoning / explicit protocol rather than a single opaque judge call).
- Uses multi-agent structure and aggregation as a mechanism to improve decision reliability.

## How is this different from GALILEO?
- D3 is primarily an *evaluation* framework for ranking/scoring model answers using adversarial debate, rather than (presumably) GALILEO’s core research contribution (method/system) if GALILEO is not centered on LLM-judge evaluation.
- D3’s novelty is the combination of (i) adversarial/advocacy roles, (ii) explicit budgeted stopping, and (iii) a probabilistic score-gap convergence model, all targeted at human-agreement in benchmarks.

## Where GALILEO is stronger / cleaner (if true)
- If GALILEO provides a single, simpler protocol with fewer moving parts, it may be easier to implement/reproduce than multi-role debate orchestration.
- If GALILEO’s evaluation approach avoids judge models entirely (or uses more direct/grounded signals), it may be less sensitive to judge prompt/role design.

## Where GALILEO is weaker / needs to improve
- If GALILEO currently uses single-shot LLM judging, D3 highlights concrete improvements: multi-advocate aggregation, iterative refinement with convergence checks, and explicit cost control.
- If GALILEO lacks explicit bias audits (position/verbosity), D3 suggests these should be part of the evaluation story.

## Action items for GALILEO (experiments / method / writing)
- [ ] Add a cost-aware evaluation section: report token cost vs agreement/quality curves (include early stopping criterion).
- [ ] Consider a “parallel advocates” ablation: k independent critiques/defenses aggregated, compared to single-judge.
- [ ] Add bias tests (position swap, verbosity control) and mitigation (anonymize, diversify roles).
- [ ] If applicable, frame evaluation as estimating a latent *gap* with uncertainty; report stability across repeated trials.

## Quotes / details to potentially cite
- Abstract protocol names and claims (verbatim-ish):
  - “Multi-Advocate One-Round Evaluation (MORE)” and “Single-Advocate Multi-Round Evaluation (SAMRE) with budgeted stopping”.
  - “probabilistic model of score gaps” with convergence / mis-ranking probability vanishing under assumptions.
  - Benchmarks: “MT-Bench, AlignBench, and AUTO-J”; improvements in “accuracy and Cohen’s kappa”; “reduced positional and verbosity biases”.
