# Agent Drift: Quantifying Behavioral Degradation in Multi-Agent LLM Systems Over Extended Interactions

- Year: 2026
- Venue: arXiv
- Authors: Abhishek Rath
- URL: https://arxiv.org/abs/2601.04170
- BibTeX key (if we add it): agentdrift_rath_2026
- Tags: multi-agent, drift, long-horizon, monitoring, stability-metrics

## One-sentence takeaway

Proposes a taxonomy and composite metric suite (Agent Stability Index) to monitor and mitigate *behavioral drift* in multi-agent LLM systems over long interaction sequences.

## What problem does it solve?

- Multi-agent LLM systems can degrade over time/turns without any explicit model update (“agent drift”), harming reliability and increasing human intervention.
- Standard evaluation is often point-in-time and does not capture long-horizon deviation in tool use, coordination, and intent.

## What is the core method / protocol?

- Defines *agent drift* and a taxonomy:
  - **Semantic drift**: deviation from original intent/spec.
  - **Coordination drift**: degradation of agreement/consensus & handoffs.
  - **Behavioral drift**: emergence of unintended strategies.
- Introduces **Agent Stability Index (ASI)**: a composite set of stability metrics across **12 dimensions** (grouped into response consistency, tool usage patterns, inter-agent coordination, behavioral boundaries).
- Uses a **simulation-based** methodology across “enterprise-style” multi-agent workflows; baseline behavior taken from early interactions (first ~20), then compares later windows to baseline.
- Mentions mitigation ideas (high-level): episodic memory consolidation, drift-aware routing, adaptive behavioral anchoring.

## What are the key metrics?

From ASI components described in the paper (examples):

- **Response consistency**: output semantic similarity; decision/reasoning pathway stability; confidence calibration drift.
- **Tool usage**: tool selection stability; tool sequencing consistency; tool parameterization drift.
- **Inter-agent coordination**: consensus agreement rate; handoff efficiency; role adherence (specialization).
- **Behavioral boundaries**: (outlined as a category; details not fully captured in this quick pass).

## What are the main results?

- Primarily a **framework + metric proposal** with simulation-based/theoretical analysis; claims that unchecked drift can materially reduce completion accuracy and raise human oversight burden.
- Does not read like a fully standardized benchmark with a broad model leaderboard (at least from abstract/intro/method sections).

## How is this similar to GALILEO?

- Shared focus on **multi-turn / long-horizon degradation** rather than single-turn accuracy.
- Emphasizes *time/trajectory-aware* evaluation and monitoring signals beyond final-task correctness.

## How is this different from GALILEO?

- Targets **multi-agent orchestration / production monitoring** (tool usage, coordination, routing), not primarily adversarial user pressure or belief/answer stability under follow-ups.
- Heavy reliance on **simulation-based workflows** and composite monitoring metrics; less emphasis on controlled perturbation protocols and clean causal comparisons.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO offers controlled perturbations (e.g., standardized follow-up attacks) and clear outcome definitions (turn-of-failure, survival-style curves), it may provide a cleaner experimental protocol than simulation-heavy, composite-index monitoring.
- GALILEO’s framing can be more directly tied to *robustness under interaction* rather than operational drift notions.

## Where GALILEO is weaker / needs to improve

- Limited coverage of **agentic tool-use drift** (tool selection/sequence/parameters) and **multi-agent coordination** metrics could leave a gap if we claim broader “agent stability” scope.
- If we don’t quantify monitoring signals, we may under-address deployment-facing concerns (routing biases, handoff inefficiency, role collapse).

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider a small “agentic drift” subsection in related work: multi-agent coordination drift and monitoring metrics (ASI-style dimensions).
- [ ] If applicable, add at least one **tool-usage stability** metric (e.g., tool sequence distance) as an auxiliary diagnostic in our experiments.
- [ ] Clarify our scope boundary: robustness under adversarial/misleading interaction vs operational drift in multi-agent systems.

## Quotes / details to potentially cite

- “This study introduces the concept of agent drift—the progressive degradation of agent behavior, decision quality, and inter-agent coherence over extended interaction sequences.”
- “We introduce the Agent Stability Index (ASI)—a novel composite metric framework quantifying drift across 12 dimensions including response consistency, tool usage patterns, reasoning pathway stability, and inter-agent agreement rates.”
