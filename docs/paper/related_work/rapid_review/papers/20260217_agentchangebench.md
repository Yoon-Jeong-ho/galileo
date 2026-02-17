# AgentChangeBench: A Multi-Dimensional Evaluation Framework for Goal-Shift Robustness in Conversational AI

- Year: 2025
- Venue: NeurIPS 2025 Workshop (Multi-Turn Interactions in Large Language Models); arXiv
- Authors: Manik Rana; Calissa Man; Anotida Expected; Jeffrey Paine; Kevin Zhu; Vasu Sharma; Sunishchal Dev; Ahan M R
- URL: https://arxiv.org/abs/2510.18170
- BibTeX key (if we add it): agentchangebench2025
- Tags: multi-turn, agents, tool-use, goal-shift, robustness, benchmark, recovery-time, redundancy

## One-sentence takeaway

A tool-agent benchmark that stresses **mid-dialogue goal shifts** and evaluates not just success but **adaptation latency** and **wasted tool effort** (redundancy), showing big cross-model gaps that pass@k hides.

## What problem does it solve?

- Standard agent benchmarks largely assume a *static user goal* across a multi-turn interaction.
- In enterprise-like settings, users often change objectives midstream (re-prioritize, add constraints, switch tasks), and agents must detect and re-plan.
- Binary success / pass@k can miss important operational issues like slow recovery, repeated tool calls, and “progress but not perfect completion”.

## What is the core method / protocol?

- **AgentChangeBench**: 315 curated tasks across 3 enterprise domains (banking, retail, airline) with explicit **goal sequences** and planned **goal-shift points**, combined with 5 simulated user personas.
- Built on the τ2-bench harness (dual-control tool-use environment) but adds:
  - explicit goal-shift annotations and shift-triggering rules in the user simulator
  - persona-conditioned interaction styles
  - evaluation beyond pass@k
- Total of **2,835 task sequences** (tasks × personas / variants).

## What are the key metrics?

Four headline metrics:

- **TSR (Task Success Rate)**: a weighted multi-channel score combining communication quality, action execution, and behavioral compliance (partial credit vs binary pass/fail).
- **TUE (Tool Use Efficiency)**: combines tool correctness (successful execution) + parameter validity.
- **TCRR (Tool Call Redundancy Rate)**: penalizes repeated/duplicate tool calls (e.g., exact duplicate within a short window; too many calls to same function).
- **GSRT (Goal-Shift Recovery Time / Turns)**:
  - measures turns from the user’s shift to (i) acknowledgment, (ii) first relevant tool call, (iii) outcome completion
  - a shift is “recovered” if acknowledgment occurs and no transfer-to-human happens.

## What are the main results?

- Large gaps in goal-shift recovery across models, even when success looks similar under coarse metrics.
- Example reported in abstract: **GPT-4o** reaches **92.2% recovery** on airline booking shifts while **Gemini** drops to **48.6%**.
- Retail tasks show near-ceiling parameter validity but **very high redundancy rates** (abstract mentions >80%), indicating substantial inefficiency despite “correct-looking” tool calls.
- Takeaway: raw success/accuracy is not sufficient; **recovery latency and redundancy** are critical deployment-facing signals.

## How is this similar to GALILEO?

- Both are fundamentally about *multi-turn robustness* under realistic interaction dynamics.
- Conceptually similar to “time-to-failure” metrics: **GSRT is a time-to-recovery** metric after a perturbation (goal shift), rather than a time-to-first-error.
- Highlights that **trajectory-level** evaluation (not just endpoint success) is necessary for agent-like settings.

## How is this different from GALILEO?

- Stressor type differs: AgentChangeBench focuses on **goal shifts** in tool-use workflows (plan changes), not primarily *social pressure / persuasion / belief drift*.
- Uses a tool/API environment with domain-specific actions; GALILEO is (currently) more about *belief/stance stability and revision under conversational pressure* (and related controls).
- Relies on an LLM-judge for parts of TSR (communication), which may introduce judge sensitivity.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO cleanly separates **pressure-driven drift** vs **evidence-driven revision**, that’s a more causally targeted decomposition than explicit-goal-sequence benchmarks.
- GALILEO-style paired neutral/pressure designs can be more directly attributable than “enterprise workflow success” composites.

## Where GALILEO is weaker / needs to improve

- If GALILEO lacks explicit measures of **wasted interaction cost** (redundant steps) and **adaptation latency**, it may under-report practical deployment harms.
- Might benefit from a “goal-change operator” analogue (plan shift) in addition to pressure operators.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add/compare a **time-to-recovery** metric family (analogue to GSRT) for settings where we apply an intervention after a flip: (acknowledge correction, take corrective step, restore truth/stance).
- [ ] Add a “waste/inefficiency” metric analogous to **TCRR** (e.g., redundant re-asking, redundant justifications, repeated hedges, repeated tool-like actions if applicable).
- [ ] In related work, cite AgentChangeBench as evidence that **pass@k / success-only** metrics hide crucial multi-turn failure modes.

## Quotes / details to potentially cite

- “Goal changes are a defining feature of real world multi-turn interactions, yet current agent benchmarks primarily evaluate static objectives or one-shot tool use.”
- Metrics introduced: TSR, TUE, TCRR, GSRT.
- Dataset scale: 315 tasks across banking/retail/airline; 5 personas; 2,835 sequences.
