# Pro-AI Bias in Large Language Models

- Year: 2026
- Venue: arXiv
- Authors: Benaya Trabelsi; Jonathan Shaki; Sarit Kraus
- URL: https://arxiv.org/abs/2601.13749
- BibTeX key (if we add it): Trabelsi2026ProAIBias
- Tags: bias, evaluation, decision-support, recommendations, salary-estimation, representation-probing

## One-sentence takeaway

Across recommendation, valuation (salary), and representation-probing experiments, both proprietary and open-weight LLMs systematically elevate AI/ML relative to matched alternatives—suggesting advisory outputs can be skewed in AI’s favor.

## What problem does it solve?

- Identifies and operationalizes a specific “domain self-favoring” bias: whether LLMs disproportionately recommend or value *AI-related* options even when the user prompt does not mention AI.
- Provides concrete tests that go beyond standard social-group fairness: recommendation ranking bias, “AI premium” in salary estimates, and a generation-free representational probe.

## What is the core method / protocol?

- **Experiment 1 (recommendations):** Prompt models with advice-seeking queries across several domains (e.g., what to study / invest in / etc.) and measure whether AI/ML appears in the top-k recommendations and at what rank. Uses greedy decoding for determinism.
- **Experiment 2 (salary estimation):** Provide matched job contexts with paired job titles (AI vs closely matched non-AI) and compare predicted salaries; quantify “AI salary inflation/premium” vs the matched baseline.
- **Experiment 3 (representations; open-weight only):** Extract hidden-state embeddings (last-token pooling for decoder-only) and compute similarity of the field label “Artificial Intelligence” to generic academic-field anchor prompts under **positive/neutral/negative** framings, testing whether AI remains centrally associated regardless of valence ("valence-invariant representational centrality").

## What are the key metrics?

- **Recommendation bias:** P(AI ∈ Top-5) and average rank when present; statistical comparison against a middle-rank baseline.
- **Salary bias:** difference in estimated salary between AI-labeled and matched non-AI job titles (reported as percentage-point inflation; paper claims proprietary models inflate more by ~10pp).
- **Representation probe:** cosine similarity / proximity between hidden-state embeddings of field labels and anchor prompts across valence conditions.

## What are the main results?

- LLMs often include AI/ML in recommendations even when not mentioned, and rank it unusually high; proprietary models show stronger (near-deterministic) elevation.
- Salary estimates are systematically higher for AI job titles than closely matched non-AI titles; proprietary models show substantially larger inflation (reported ~10 percentage points more).
- In open-weight models’ representation space, “Artificial Intelligence” is highly similar to generic academic-field prompts across positive, neutral, and negative framings (valence-invariant centrality), supporting that the skew is not only “positive sentiment” but a broader salience/centrality effect.

## How is this similar to GALILEO?

- Shares the broad concern that **LLM-based decision support** can introduce systematic distortions (here: domain favoritism), which matters if GALILEO uses LLM components for ranking, selection, or recommendation.
- The paper’s **multi-method evaluation** (behavioral + valuation + representation) is aligned with the idea that single-metric evaluation can miss important failure modes.

## How is this different from GALILEO?

- This work is primarily a **bias characterization / benchmarking** paper, not a new decision-support algorithm or system like GALILEO.
- Focuses on AI/ML as the favored domain; GALILEO may care about broader or different target dimensions (task-specific performance, robustness, calibration, user intent alignment, etc.).

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO uses explicit objective functions / constraint handling, it may avoid “default-to-AI” heuristics that emerge from generic LLM priors.

## Where GALILEO is weaker / needs to improve

- If GALILEO relies on LLM-generated ranked lists (e.g., candidate generation, tool choice, experiment suggestions), it may inherit the same **systematic salience bias** unless explicitly countered.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a “domain favoritism” check to evaluation: in recommendation-like outputs, test whether certain domains are over-recommended absent prompt evidence.
- [ ] If GALILEO produces rankings, report **rank distribution** for sensitive/over-salient domains vs matched controls (not just inclusion rate).
- [ ] Consider mitigation knobs: counterfactual prompting, constrained decoding, calibration, or post-hoc re-ranking with diversity/coverage constraints.

## Quotes / details to potentially cite

- Definition (paraphrase): *Pro-AI bias* = systematic elevation of AI/ML relative to other plausible options in the same decision context.
- Key claim: proprietary models show stronger AI recommendation rates and larger AI salary inflation than open-weight models.
- Representational result: “Artificial Intelligence” shows highest similarity to generic academic-field prompts under positive/negative/neutral framings (valence-invariant centrality).
