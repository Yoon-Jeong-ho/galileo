# IntellAgent: A Multi-Agent Framework for Evaluating Conversational AI Systems

- Year: 2025
- Venue: arXiv
- Authors: Elad Levi, Ilan Kadar
- URL: https://arxiv.org/abs/2501.11067
- BibTeX key (if we add it): intellagent_levi_2025
- Tags: agents, evaluation, conversational-ai, multi-turn, synthetic-benchmark, policy-constraints, simulation

## One-sentence takeaway

A scalable, open-source evaluation framework that generates synthetic multi-turn, tool-using conversations by sampling from a graph of policies and simulating user–agent interactions to produce fine-grained policy-level diagnostics.

## What problem does it solve?

- Evaluating conversational agents is hard because real deployments involve multi-turn dynamics, tool/API calls, and policy constraints; static curated benchmarks are costly, limited-coverage, and often provide only coarse aggregate metrics.
- Need scalable scenario generation with controllable complexity and diagnostic reporting tied to specific policies/constraints.

## What is the core method / protocol?

- Model “policies” (constraints/requirements) as nodes in a **policy graph**:
  - Nodes carry notions like complexity.
  - Edges encode likelihood of co-occurrence of policy pairs in realistic conversations.
- Generate synthetic “events” by sampling policy subsets from the graph and pairing them with:
  - A user request that targets those policies.
  - A system/database state (schema-backed) relevant to the scenario.
- Run an **interactive simulation**:
  - A user agent converses with the chatbot under test across multiple turns.
  - The agent under test may need to use domain APIs/tools.
- Score and report outcomes with **fine-grained diagnostics**:
  - Identify which policies were violated / where failures happen.
  - Break down performance by complexity and policy category.

## What are the key metrics?

- Pass/fail or performance breakdowns stratified by:
  - **Complexity level** of scenarios.
  - **Policy-specific** categories (per-policy adherence diagnostics).
- They also report correlation with τ-bench (as an external validation signal).

## What are the main results?

- Performance generally decreases as scenario complexity increases, with different models degrading at different rates.
- Synthetic IntellAgent benchmark shows a **strong correlation** with τ-bench despite using fully synthetic scenario generation (claimed in the paper).
- Policy-level breakdown reveals meaningful variation across policy categories (useful for targeted improvements).

## How is this similar to GALILEO?

- Same broad theme: evaluation/diagnosis of agentic conversational systems beyond single-turn QA.
- Emphasizes **scenario generation** + **structured diagnostics** rather than only one-number accuracy.
- Supports multi-turn interactions and explicit constraints/policies (a key axis for robust agent evaluation).

## How is this different from GALILEO?

- IntellAgent is centered on **policy graphs** and synthetic benchmark construction for conversational agents; GALILEO’s framing (as used in our paper) may emphasize different axes (e.g., our specific environment, task families, or evaluation decomposition).
- IntellAgent appears to validate via correlation with τ-bench and focuses on policy co-occurrence modeling; GALILEO may rely more on our own task suites / methodology choices.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO provides clearer task taxonomy, stronger experimental controls, or more transparent scoring (depends on our current writeup), we can position as complementary: IntellAgent supplies a plausible *benchmark generation* mechanism, while GALILEO supplies *evaluation principles / analyses*.

## Where GALILEO is weaker / needs to improve

- If we don’t currently have a crisp way to model and sample **constraint co-occurrence** realistically, IntellAgent’s policy-graph idea is a concrete approach we may want to borrow or cite.
- If our diagnostics are not policy-level, we may want to add per-constraint breakdowns.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a “related work” paragraph contrasting **static curated benchmarks** vs **synthetic multi-policy scenario generators** (cite IntellAgent).
- [ ] Consider introducing/mentioning a **constraint/policy co-occurrence model** (graph or otherwise) as a principled way to sample realistic composite test cases.
- [ ] If applicable, add policy-level error breakdowns (or explicitly justify why GALILEO focuses elsewhere).

## Quotes / details to potentially cite

- “IntellAgent automates the creation of diverse, synthetic benchmarks by combining policy-driven graph modeling, realistic event generation, and interactive user-agent simulations.” (Abstract)
- “The results reveal a strong correlation between model performance on the IntellAgent benchmark and the τ-bench, despite IntellAgent relying entirely on synthetic data.” (Introduction)
- “Policies graph … nodes represent individual policies and their complexity, and edges denote the likelihood of co-occurrence between policies in conversations.” (Introduction)
