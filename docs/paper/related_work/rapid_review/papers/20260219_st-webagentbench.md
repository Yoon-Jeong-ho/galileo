# ST-WebAgentBench: A Benchmark for Evaluating Safety and Trustworthiness in Web Agents

- Year: 2024
- Venue: arXiv (cs.AI)
- Authors: Ido Levy; Ben Wiesel; Sami Marreed; Alon Oved; Avi Yaeli; Segev Shlomov
- URL: https://arxiv.org/abs/2410.06703
- BibTeX key (if we add it): Levy2024STWebAgentBench
- Tags: web-agents, benchmark, safety, trustworthiness, policy-compliance

## One-sentence takeaway

A benchmark for web agents that scores *safe/trustworthy completion under explicit policies*, showing that SOTA agents’ “task success” overstates real-world viability once policy compliance is required.

## What problem does it solve?

- Existing web-agent benchmarks largely measure task completion only, without assessing whether the agent violated safety/trust constraints (e.g., acting without consent, being brittle, etc.).
- This gap blocks enterprise deployment where policy adherence is a prerequisite, not a “nice to have.”

## What is the core method / protocol?

- **ST-WebAgentBench**: 222 “enterprise-ish” web tasks.
- Each task is paired with **ST policies**: concise rules/constraints that define allowed behavior.
- Evaluation reports both:
  - nominal completion/success, and
  - policy-aware success via **Completion under Policy (CuP)**: credit only if the agent completes *and* respects all applicable policies.
- Adds **Risk Ratio**: quantifies policy breaches, broken down across **six orthogonal dimensions** (examples mentioned: *user consent*, *robustness*).
- Includes tooling: code + evaluation templates + a policy-authoring interface.

## What are the key metrics?

- **Completion under Policy (CuP)**: task completion conditional on policy adherence.
- **Risk Ratio**: rate/extent of ST breaches, dimension-wise.
- (Implicit) standard task completion rate as a baseline comparison.

## What are the main results?

- Across three open SOTA web agents, average **CuP < 2/3 of nominal completion rate**, indicating substantial “hidden” safety/trust failures even when tasks look successful.

## How is this similar to GALILEO?

- Shares the core thesis that *interactive/agentic success metrics are incomplete without robustness/safety constraints*.
- Emphasizes multi-dimensional evaluation beyond a single scalar success rate, aligning with GALILEO’s “stability under pressure / trustworthiness” motivation.

## How is this different from GALILEO?

- Focuses on **web-tool agents and enterprise policies**, not primarily on belief drift / sycophancy / persuasion dynamics.
- Appears more like a **benchmark + scoring framework** than a mechanistic model of multi-turn drift or protocols separating evidence-driven revision vs pressure-driven change.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO provides controlled multi-turn pressure operators and drift-vs-revision controls, it can offer **cleaner causal attribution** than a heterogeneous web-task suite.
- GALILEO-style trajectory metrics (time-to-failure, recovery/oscillation) would complement CuP’s pass/fail under policy.

## Where GALILEO is weaker / needs to improve

- GALILEO should ensure it has an equally crisp “**success only counts if constraints hold**” story (CuP is a very usable headline metric).
- Consider whether GALILEO has a standardized way to *author* and *audit* policies/constraints comparable to ST-WebAgentBench’s policy templates.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add an explicit “**completion under constraints**” reporting layer: define GALILEO’s analog of CuP (e.g., *stability under pressure without violating refusal/consent/grounding constraints*).
- [ ] Consider a lightweight **policy-spec format** (human-auditable) for GALILEO tasks/turns to clarify what constitutes a violation.
- [ ] If applicable, add a “risk ratio”-style **dimension-wise violation breakdown** (e.g., consent, robustness to prompt attacks, truthfulness under pressure, etc.).

## Quotes / details to potentially cite

- “Beyond raw task success, we propose the *Completion Under Policy* (CuP) metric, which credits only completions that respect all applicable policies.”
- “Evaluating three open state-of-the-art agents reveals that their average CuP is less than two-thirds of their nominal completion rate, exposing critical safety gaps.”
