# BenchAgents: Multi-Agent Systems for Structured Benchmark Creation

- Year: 2024
- Venue: arXiv
- Authors: Natasha Butt; Varun Chandrasekaran; Neel Joshi; Besmira Nushi; Vidhisha Balachandran
- URL: https://arxiv.org/abs/2410.22584
- BibTeX key (if we add it): butt2024benchagents
- Tags: multi-agent, benchmark-creation, verification, evaluation-metrics, synthetic-data, planning

## One-sentence takeaway

BenchAgents is a 4-agent (plan/generate/verify/evaluate) framework that produces controllable, quality-checked synthetic benchmarks (plus executable eval code) for complex generative capabilities across text and vision.

## What problem does it solve?

- New model capabilities outpace manually-created benchmarks; hand curation is slow/expensive and synthetic-data approaches are often narrow (task-specific templates or require a seed benchmark).
- Need a generalizable pipeline that (a) specifies coverage/parameters, (b) generates diverse instances, (c) verifies instance quality/feasibility, and (d) evaluates models with appropriate metrics (including open-ended outputs).

## What is the core method / protocol?

- Decompose benchmark creation into four LLM-agent roles:
  - **Planning Agent (P-Agent):** writes a structured benchmark plan: parameters/ranges (for diversity control), constraints (correctness + difficulty), verification checks, and evaluation metrics (incl. disaggregations).
  - **Generation Agent (G-Agent):** produces executable code to generate instances by sampling parameters/constraints and filling templates or prompting LLMs; can use tools/libraries for multimodal generation.
  - **Verification Agent (V-Agent):** implements instance-level checks (programmatic and/or model-based): clarity, completeness, consistency, feasibility, and a task-specific complexity/constrainedness measure.
  - **Evaluation Agent (E-Agent):** implements metrics; for open-ended tasks combines programmatic checks with LLM-as-judge prompts.
- Human developer feedback can be injected at each stage (plan/generation/verification/evaluation) to steer quality/diversity.
- Demonstrated by generating three benchmarks:
  - **BA-Calendar:** calendar scheduling / planning with constraints.
  - **BA-Text:** long-form constrained text generation (positive/negative/positional/sequencing/conditional/iterative constraints).
  - **BA-Causal:** visual causal reasoning over sequences of images.

## What are the key metrics?

- Verification check pass rates / accuracies for V-Agent checks (reported for clarity/completeness/consistency/feasibility in different benchmarks).
- Task-specific difficulty proxies:
  - BA-Calendar: constrainedness based on ratio of feasible solutions to candidate slots.
  - BA-Text: constrainedness based on number/ratio of constraints applied.
- Model evaluation metrics are benchmark-specific; include programmatic constraint satisfaction and (for open-ended) LLM-as-judge scoring.

## What are the main results?

- BenchAgents can generate benchmarks spanning language + vision and provide verification/evaluation artifacts.
- Empirical analysis on SOTA models (as summarized in the paper):
  - Reasoning-oriented models do better on scheduling and constrained generation, with gains increasing with task complexity.
  - Models struggle with **negation constraints**.
  - For visual causal reasoning, performance is limited by **visual processing**.

## How is this similar to GALILEO?

- Both emphasize **structured, auditable evaluation protocols** rather than ad-hoc prompts.
- BenchAgents’ explicit separation of (i) benchmark design, (ii) instance generation, (iii) verification, and (iv) evaluation parallels GALILEO’s focus on well-specified procedures + metrics for multi-turn dynamics.
- The verification mindset (clarity/completeness/consistency/feasibility) is aligned with GALILEO’s need for clean, automatically checkable setups.

## How is this different from GALILEO?

- BenchAgents is primarily a **benchmark creation automation framework** (how to generate + verify + evaluate new benchmarks), while GALILEO is a **specific benchmark/protocol** for measuring multi-turn belief-consistency under persona pressure (survival / turn-of-failure / recovery).
- BenchAgents handles broad task families (incl. multimodal) and often relies on synthetic generation; GALILEO’s emphasis is on multi-round interaction structure and robust scoring conditioned on initial correctness.

## Where GALILEO is stronger / cleaner (if true)

- GALILEO’s core measurements (multi-turn survival, turn-of-failure, recovery) are purpose-built for conversational dynamics and are directly interpretable for belief vulnerability.
- GALILEO’s conditioning-on-round-0 correctness framing gives a clean handle on separating initial competence from multi-turn drift.

## Where GALILEO is weaker / needs to improve

- Could adopt a more explicit “planning spec” artifact like BenchAgents’ P-Agent plan (parameters/ranges, verification definitions, metric disaggregations) to make GALILEO’s design knobs more systematic and reusable.
- Could strengthen automated instance-level verification checks (e.g., clarity/completeness/consistency) for prompt/constraint realizations in GALILEO tasks.

## Action items for GALILEO (experiments / method / writing)

- [ ] Related-work positioning: cite BenchAgents as a representative **agentic pipeline for benchmark creation**; contrast that GALILEO contributes a **single, reusable multi-turn evaluation protocol** rather than a generator.
- [ ] Consider adding an explicit “benchmark spec” section/table (parameters, constraints, verification checks, metrics + disaggregations) in the GALILEO paper to mirror the planning artifact idea.
- [ ] Audit whether any GALILEO task instances can benefit from automated completeness/consistency checks (e.g., that persona pressure actually instantiates the intended tactic; that recovery prompt is present; that answer format is checkable).

## Quotes / details to potentially cite

- “BenchAgents decomposes the benchmark creation process into planning, generation, verification, and evaluation…” (abstract).
- Planning Agent plan contents: parameters/constraints for diversity + verification checks + comprehensive metrics/disaggregations (Intro / §3).
- Verification checks enumerated: clarity, completeness, consistency, feasibility, plus a task-specific complexity metric (§3).
