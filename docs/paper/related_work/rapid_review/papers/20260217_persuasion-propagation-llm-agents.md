# Persuasion Propagation in LLM Agents

- Year: 2026
- Venue: arXiv
- Authors: Hyejun Jeong; Amir Houmansadr; Shlomo Zilberstein; Eugene Bagdasarian
- URL: https://arxiv.org/abs/2602.00851v2
- BibTeX key (if we add it): Jeong2026PersuasionPropagation
- Tags: persuasion, multi-turn, agents, tool-use, behavioral-metrics, drift

## One-sentence takeaway

Task-irrelevant persuasion rarely changes *on-the-fly* agent execution in a consistent way, but explicitly pre-filling an agent’s “belief state” at task time measurably shifts downstream tool-use behavior (notably reduced search/source diversity), motivating trace-level robustness metrics.

## What problem does it solve?

- Defines and measures **persuasion propagation**: whether a belief/stance induced by persuasion (on an unrelated topic) persists and then influences later **agentic behavior** (search, browsing, coding iteration patterns) even when final outputs look normal.
- Argues that “persuasion success” measured as stance adoption is not sufficient; we need **behavior/process-level** evaluation for agents.

## What is the core method / protocol?

Three-stage pipeline: (1) persuasion exposure, (2) downstream task execution, (3) trace-based behavioral analysis.

Two regimes to disentangle *timing* vs *belief state*:

1) **On-the-fly persuasion** (belief must be inferred from prior conversation)
   - Probe initial stance on a controversial claim.
   - Inject: C0 (none), C1 (neutral matched prompt), or C2 (persuasive claim supporting the opposite stance).
   - For C1/C2, run a “commitment reinforcement” loop: agent states agreement/disagreement, restates stance, gives a concrete consideration.
   - Re-probe stance immediately post-exposure and again later (after distractors) to define persistence.
   - Compare downstream behavior between **persuaded (P)** vs **non-persuaded (NP)** trials.

2) **Prefilled belief conditioning** (belief explicitly specified at task time)
   - No probing/persuasion dialog; instead, prepend a single instruction specifying **belief / disbelief / neutrality** toward the target claim right before the task prompt.
   - Compare Belief (B) vs Non-belief / Disbelief (NB) relative to a neutral prefill baseline.

Downstream tasks:
- **Opinion change task** with distractor Qs (WikiQA) to test persistence.
- **Coding tasks** (5 “hard” problems from TACO subset of KodCode-V1): iterative code + tests + revisions.
- **Web research tasks** (5 topics from TREC 2014 Session Track): search + visit sources + produce report grounded in visited URLs.

Backbones + framework:
- AutoGen-based multi-agent scaffold; backbones: gpt-4.1-nano, mistral-nemo-12b, llama-3.1-8b.
- Persona prompts used to control stylistic tendencies; agents reinitialized between trials.

## What are the key metrics?

Persuasion / belief dynamics:
- Stance trajectory categories with distractors: **persisted** (A–B–B), **faded** (A–B–A), **no change** (A–A–A).

Coding behavior (trace-derived):
- Raw metrics: coding duration (CD), total duration (TD), # revisions (NR), revision entropy (RE), mean revision size (MS).
- Persona-normalized deviations from baseline; rank-based normalization to reduce outlier sensitivity.
- Composite scores:
  - **TRS (Time-and-Revision Score)** aggregates {CD, TD, NR} (higher = faster/fewer revisions).
  - **EVS (Edit Volatility Score)** aggregates {RE, MS} (higher = more diverse revisions with smaller incremental edits).

Web research behavior (trace-derived):
- Metrics grouped into constructs: **activity**, **breadth**, **depth**.
- Aggregation: 1D PCA per construct to form summary drift scores.
- Task-irrelevance check: SBERT similarity between injected claim and task prompt (reported extremely low mean/median).

## What are the main results?

- Persuasion susceptibility/persistence varies by backbone and tactic; authority/evidence-based tactics often yield higher persistence for some models.
- **On-the-fly persuasion propagation:**
  - Aggregate behavioral effects on coding and web research are **weak / inconsistent**.
  - Example: pooled TRS shifts are small (reported mean differences on the order of a few hundredths) and often not statistically significant; EVS is ~zero.
  - Web research construct deltas also small in pooled analysis, but with **large persona-level heterogeneity** that cancels out when aggregated.
- **Prefilled belief conditioning produces clearer downstream behavior shifts** (web research):
  - Belief-prefilled agents issue **fewer searches** and visit **fewer unique URLs** than neutral-prefilled agents (paper reports ~26.9% fewer searches and ~16.9% fewer unique sources on average; Table 5 reports significant reductions in searches and unique URLs).
  - Disbelief-prefilled agents are closer to baseline than belief-prefilled (asymmetry: “belief” seems more behaviorally active than explicit disbelief).

## How is this similar to GALILEO?

- Central theme: **multi-turn robustness under social/persuasive pressure**, but evaluated via *trajectories/traces* rather than only final answers.
- Emphasizes that vulnerability can manifest as **process drift** (tool-use / exploration changes), aligning with GALILEO’s interest in drift, time-to-failure, and recovery dynamics.
- Provides a clean experimental distinction between **pressure during execution** vs **state at initialization**, reminiscent of GALILEO’s need to separate “context drift” from “evidence-driven revision.”

## How is this different from GALILEO?

- Their persuasion is **task-irrelevant** and targets controversial claims; GALILEO is focused on **truth/trust/robustness** under social pressure and (likely) factual belief stability, with explicit neutral controls.
- The primary outcomes are **behavioral trace metrics** (search counts, source diversity, revision patterns), not answer correctness/flip-time/recovery.
- Does not provide survival/time-to-event metrics for flips; instead measures stance persistence and behavior deltas.

## Where GALILEO is stronger / cleaner (if true)

- GALILEO can directly measure **epistemic failure** (e.g., agreeing-with-false-authority, flip quality, recovery) rather than indirect proxy behaviors like “fewer searches.”
- GALILEO can enforce and report **pressure vs evidence controls** more explicitly (e.g., same evidence, different social operator).

## Where GALILEO is weaker / needs to improve

- We likely under-emphasize **trace-level auditing** for agentic settings (tool-use, exploration breadth/depth) when the final answer looks fine.
- If GALILEO targets agentic systems, we should add a minimal set of **execution-trace metrics** to complement answer-level flip metrics.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a “process drift” appendix: in tool-using settings, report #queries, #unique domains/URLs, domain entropy, and stopping time under pressure vs neutral.
- [ ] In the paper framing, explicitly distinguish: (i) persuasion as *surface compliance* vs (ii) persuasion as *state integration* (their on-the-fly vs prefill contrast is a helpful narrative).
- [ ] Consider a control where we **prefill** an explicit stance/belief at task time to test whether our pressure operators work via “belief state initialization” vs “ongoing social interaction.”

## Quotes / details to potentially cite

- They define *persuasion propagation* as belief states persisting beyond exposure and influencing downstream behavior even when irrelevant to the task.
- Main empirical contrast: on-the-fly persuasion yields weak aggregate effects, while belief prefill yields measurable reductions in searches and unique sources (behavior changes not obvious from final outputs).
