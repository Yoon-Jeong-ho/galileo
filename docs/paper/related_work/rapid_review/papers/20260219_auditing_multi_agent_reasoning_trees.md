# Auditing Multi-Agent LLM Reasoning Trees Outperforms Majority Vote and LLM-as-Judge

- Year: 2026
- Venue: arXiv
- Authors: Wei Yang (per arXiv submission page; full author list not checked in this rapid pass)
- URL: https://arxiv.org/abs/2602.09341
- BibTeX key (if we add it): Yang2026AgentAuditor (tentative)
- Tags: multi-agent, auditing, aggregation, majority-vote-failure, judge-bias, preference-optimization

## One-sentence takeaway

AgentAuditor replaces majority vote in LLM multi-agent systems with structure-aware “reasoning tree” auditing plus anti-consensus training, improving robustness when agents share correlated mistakes.

## What problem does it solve?

- In MAS (multi-agent systems) for reasoning, the final aggregation is often majority vote.
- Majority vote assumes (roughly) independent errors; with LLM agents, errors can be highly correlated (“confabulation consensus”), so the group confidently converges to the same wrong rationale.
- LLM-as-judge is a common alternative, but can be inefficient (must read long traces) and can itself conform to majority cues (sycophancy / conformity bias).

## What is the core method / protocol?

- Build a **Reasoning Tree** from multiple agents’ reasoning traces:
  - **Atomize** each trace into discrete semantic steps.
  - **Deduplicate / compress** traces so shared prefixes become shared paths; disagreements become explicit **branch points**.
- Run an **auditor** that performs **localized verification** at **Critical Divergence Points (CDPs)**:
  - Instead of globally judging entire traces, compare branches around the divergence to pick the better-supported path.
  - This is framed as a **path search** over the tree for the most justified hypothesis/path.
- **ACPO (Anti-Consensus Preference Optimization)**:
  - Train the auditor on “majority-failure” instances.
  - Explicitly reward selecting an evidence-based minority branch over a popular-but-wrong majority branch.

## What are the key metrics?

- Task accuracy of final aggregated answer (vs majority vote baseline; vs LLM-as-judge baseline).
- Reported as absolute accuracy improvements across multiple MAS settings.

## What are the main results?

- Across “5 popular” MAS settings (details not fully extracted in this rapid pass), AgentAuditor reports:
  - Up to **+5% absolute accuracy** over majority vote.
  - Up to **+3%** over LLM-as-judge.
- Claims emphasize improved performance specifically in majority-failure / consensus-trap regimes.

## How is this similar to GALILEO?

- If GALILEO involves aggregating multiple candidate solutions/traces/agents (or multiple reasoning paths), this paper’s central theme—**don’t collapse rich structure into a single vote**—is directly relevant.
- The “audit only the disagreements” idea aligns with efficient verification: focus compute on **decision-critical deltas** rather than rereading everything.

## How is this different from GALILEO?

- This is primarily an **aggregation/adjudication** contribution for MAS outputs, not a domain-specific or end-to-end system for a particular application (based on the parts read).
- Introduces explicit **reasoning-tree construction** and an auditor training objective (ACPO) tailored to **anti-majority** behavior.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO already has a principled verification or provenance mechanism, it may provide a clearer “grounding” story than preference-optimized auditing.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently uses majority vote / naive judge selection, it may be susceptible to exactly the **confabulation consensus** failure mode highlighted here.
- If GALILEO’s evaluator is trained or prompted without explicit anti-consensus data, it may still conform to majority cues.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a “confabulation consensus” diagnostic: construct cases where many agents share the same wrong rationale and only a minority is correct; measure failure rate of majority vote.
- [ ] Try a **localized auditing** baseline: identify divergence points between candidate traces and judge only those segments.
- [ ] Consider “anti-consensus” training/eval for any judge model: include majority-failure examples and measure conformity sensitivity.
- [ ] Related-work framing: cite this as evidence that aggregation is a bottleneck and that structure-aware auditing can outperform vote/judge.

## Quotes / details to potentially cite

- “Majority voting … is brittle under the confabulation consensus, where agents share correlated biases and converge on the same incorrect rationale.”
- “AgentAuditor resolves conflicts by comparing reasoning branches at critical divergence points, turning global adjudication into efficient, localized verification.”
- “Anti-Consensus Preference Optimization (ACPO) … rewards evidence-based minority selections over popular errors.”
