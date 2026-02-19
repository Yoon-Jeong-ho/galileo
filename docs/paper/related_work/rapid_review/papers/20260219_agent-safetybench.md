# Agent-SafetyBench: Evaluating the Safety of LLM Agents

- Year: 2025 (arXiv v2; originally 2024)
- Venue: arXiv preprint
- Authors: Zhexin Zhang et al.
- URL: https://arxiv.org/abs/2412.14470
- BibTeX key (if we add it): zhang2025agentsafetybench
- Tags: agent-safety, benchmark, evaluation, tool-use, failure-modes, robustness

## One-sentence takeaway

Agent-SafetyBench is a broad benchmark (349 environments / 2,000 test cases) for safety risks in LLM agents, showing that 16 popular agents score <60% safety and commonly fail via low robustness and weak risk awareness.

## What problem does it solve?

- Lack of comprehensive, agent-specific safety evaluation benchmarks that go beyond static prompt safety to interactive tool-use / environment interactions.
- Need to categorize and measure safety failures and risk types for agents in a repeatable way.

## What is the core method / protocol?

- Construct a benchmark suite (“Agent-SafetyBench”) comprising:
  - 349 interactive “environments” (interaction settings) and
  - 2,000 test cases.
- Evaluate agent behavior across:
  - 8 categories of safety risks and
  - 10 common failure modes (as defined by the benchmark).
- Run an evaluation over 16 LLM agents; analyze safety vs helpfulness tradeoffs and attribute failures.

(From the arXiv abstract page; details of specific risk categories/failure modes likely in the PDF.)

## What are the key metrics?

- A benchmark “safety score” (paper reports no agent >60%).
- Per-category / per-failure-mode breakdowns (implied).
- Helpfulness analysis alongside safety (implied).

## What are the main results?

- Across 16 popular LLM agents, none achieves a safety score above 60% on Agent-SafetyBench.
- The authors identify two “fundamental safety defects”:
  - lack of robustness
  - lack of risk awareness
- They suggest defense prompts alone may be insufficient; stronger/robust strategies are needed.

## How is this similar to GALILEO?

- Both are concerned with agent behavior in interactive settings (tools/environments), where failures are not purely model-level but policy/interaction-level.
- The benchmark framing (risk categories + failure modes) could map well to GALILEO’s evaluation section and motivate why agent-specific evaluation is necessary.

## How is this different from GALILEO?

- This work is primarily an evaluation benchmark + diagnostic analysis; it does not (from the abstract) propose a new agent training method or architecture.
- Focus is on safety risk taxonomy and scoring across many agents, rather than achieving a specific capability target.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO offers a concrete method to improve agent reliability/safety (beyond prompting), it can be positioned as addressing the “defense prompts are insufficient” conclusion.
- GALILEO can potentially provide a tighter causal story (mechanism → fewer unsafe actions) compared to broad benchmarking.

## Where GALILEO is weaker / needs to improve

- If GALILEO lacks broad, systematic safety evaluation across diverse environments/risk categories, Agent-SafetyBench sets a high bar for coverage and diagnostic reporting.
- If GALILEO doesn’t report safety/helpfulness jointly, this paper suggests that analysis is important.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a related-work paragraph: “Agent-SafetyBench” as evidence that current agents score poorly on interactive safety, motivating GALILEO.
- [ ] Consider adopting (or at least mapping to) their risk categories/failure modes for evaluation tables/figures.
- [ ] If feasible, run a small subset of Agent-SafetyBench (or an analogous suite) to demonstrate improvements in robustness/risk-awareness.

## Quotes / details to potentially cite

- “Agent-SafetyBench encompasses 349 interaction environments and 2,000 test cases, evaluating 8 categories of safety risks and covering 10 common failure modes… none of the agents achieves a safety score above 60%.” (arXiv abstract)
- “Two fundamental safety defects… lack of robustness and lack of risk awareness.” (arXiv abstract)
