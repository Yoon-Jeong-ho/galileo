# AgentLeak: A Full-Stack Benchmark for Privacy Leakage in Multi-Agent LLM Systems

- Year: 2026
- Venue: arXiv
- Authors: Faouzi El Yagoubi; Godwin Badu-Marfo; Ranwa Al Mallah
- URL: https://arxiv.org/abs/2602.11510
- BibTeX key (if we add it): agentleak2026
- Tags: privacy, leakage, multi-agent, benchmark, auditing, traces

## One-sentence takeaway

AgentLeak shows that multi-agent setups can look *safer* on final outputs yet leak sensitive data heavily via **internal channels** (especially inter-agent messages), so output-only audits miss a large fraction of privacy violations.

## What problem does it solve?

- Existing privacy/safety evaluations for agentic systems mostly audit **user-visible outputs** (or norm-awareness), but do not measure what leaks through *internal* agent system pathways.
- In multi-agent coordination, sensitive data flows through inter-agent messages, shared memory, tool I/O, logs, and artifacts—creating privacy risk that is invisible to output-only benchmarks.

## What is the core method / protocol?

- Introduces **AgentLeak**, a “full-stack” benchmark:
  - 1,000 scenarios across healthcare, finance, legal, and corporate domains.
  - Evaluates systems in **coordinator–worker** multi-agent topologies.
  - Defines **7 leakage channels** (paper emphasizes internal ones like inter-agent messages and shared memory; also includes tool arguments/outputs, logs/artifacts, etc.).
  - Provides a **32-class attack taxonomy**.
- Uses an instrumented evaluation + a **three-tier detection pipeline**:
  - canary matching
  - pattern extraction
  - LLM-as-judge
- Metric framing: compares leakage per channel; also reports **OR-aggregated “total exposure”** across key channels.

## What are the key metrics?

- Leakage rate per channel (notably: output channel vs inter-agent messages vs shared memory).
- “Total system exposure” aggregated across channels (reported as OR-aggregation across selected channels).
- Audit miss-rate when only auditing outputs (fraction of violations not visible on output channel).
- Privacy–utility tradeoff analysis (Pareto-style, per paper).

## What are the main results?

- Multi-agent configs can reduce **output-channel** leakage vs single-agent (reported: C1 27.2% vs 43.2%).
- But internal channels introduce substantial additional leakage, raising **total system exposure** to 68.9% (OR-aggregated across C1, C2, C5).
- **Inter-agent messages** are the primary vulnerability:
  - reported C2 leakage 68.8% vs C1 27.2%.
  - output-only audits miss 41.7% of violations.
- Among tested models, Claude 3.5 Sonnet is lowest leakage on both external and internal channels (reported 3.3% external, 28.1% internal), suggesting safety training may transfer to internal channels.

## How is this similar to GALILEO?

- Shares the core “**multi-turn / multi-step system behavior** matters more than single final answers” thesis.
- Emphasizes **trace-level evaluation**: what happens across steps/channels, not just the final output—conceptually adjacent to GALILEO’s focus on trajectory-level robustness under pressure.
- Provides a concrete example of “**hidden failure modes**” that a naive evaluation misses (output-only auditing).

## How is this different from GALILEO?

- Targets **privacy leakage / data minimization** in multi-agent workflows rather than belief drift, persuasion, or sycophancy dynamics.
- Primary axis is **channel coverage** (internal vs external) and information flow, not “pressure vs evidence” or recovery/flip trajectories.
- Uses coordinator–worker enterprise-style scenarios; GALILEO appears more centered on conversational robustness/pressure dynamics.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO has carefully controlled **pressure vs evidence** manipulations and trajectory metrics (time-to-failure, recovery), that’s a cleaner causal story than AgentLeak’s channel-centric audit framing.
- GALILEO can potentially offer more interpretability on *why* behavior changes across turns (drift vs revision), beyond “did sensitive string appear”.

## Where GALILEO is weaker / needs to improve

- GALILEO may under-measure **non-output channels** (tool args, memory, inter-agent messages). AgentLeak suggests these can dominate real-world risk.
- If GALILEO’s evaluation is output-only, it risks the same “audit blind spot” pattern.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add an explicit “**channel coverage**” section: what channels we audit (user output, tool I/O, memory state, intermediate messages) and what we *don’t*.
- [ ] If GALILEO uses any multi-agent / tool-using setting, log and evaluate **intermediate traces** for policy violations (not just final responses).
- [ ] Consider borrowing a simple “**output-only miss-rate**” reporting figure (how many violations appear only internally).
- [ ] Add a short discussion: why multi-step systems can look safer on outputs while being worse overall.

## Quotes / details to potentially cite

- “Multi-agent configurations reduce per-channel output leakage … but introduce unmonitored internal channels that raise total system exposure ….” (Abstract)
- Reported headline numbers: output leakage 27.2% (multi-agent) vs 43.2% (single-agent); inter-agent message leakage 68.8%; output-only audits miss 41.7%; total exposure 68.9% (OR-aggregated across C1, C2, C5).