# Towards Trustworthy Multi-Turn LLM Agents via Behavioral Guidance

- Year: 2025
- Venue: AAAI 2026 Workshop on Trust and Control in Agentic AI (TrustAgent) (arXiv preprint)
- Authors: Gonca Gürsun
- URL: https://arxiv.org/html/2512.11421v1 (abs: https://arxiv.org/abs/2512.11421)
- BibTeX key (if we add it): Gursun2025BehavioralGuidanceTrustworthy
- Tags: agents, multi-turn, reliability, verification, constraints, RL-loop, profiling

## One-sentence takeaway

Proposes an RL-style observation–action–reward loop for LLM agents augmented with (i) an LLM task-profiler, (ii) a learned “rule bank” of verifiable observation→action mappings, and (iii) a constraint-enforcing generation layer (validation + deterministic fallback), improving stability on Wordle/Guess-My-Number.

## What problem does it solve?

- Multi-turn LLM agents often behave inconsistently across turns and violate task constraints even when they can restate the constraints in natural language.
- Lack of *verifiability*: it is hard to audit why an action was selected and whether it matches learned experience.
- Lack of *reliability*: output compliance and performance varies widely across trajectories.

## What is the core method / protocol?

- Formalize the task as an RL-like environment interface with explicit **observations**, **actions**, and **rewards**.
- Add three modules around a “prompting backbone”:
  - **Task profiler** (LLM function): inspects environment/task variables and selects a reasoning strategy + generation strategy (e.g., short-horizon vs long-horizon; direct generation vs deterministic enumeration).
  - **Reasoning module**: mines past successful trajectories to extract reusable, interpretable rules (“if observation features, then best next action”), stores them in a **Rule Bank** with usage/success stats; uses profiler to choose temporal windowing (single-step vs multi-step/cumulative).
  - **Generation module**: enforces constraint-compliant outputs via validity checking; if invalid, falls back to structured procedures (e.g., deterministic enumeration / code-based synthesis) to produce a valid action.
- Co-evolution across epochs: profiler may update, rules get refined, generation adapts to updated rules.

## What are the key metrics?

- Average task reward per epoch (with confidence intervals).
- **Reasoning consistency ratio**: fraction of turns where an applicable learned rule is correctly invoked.
- **Constraint-compliance ratio** (Wordle): fraction of turns with outputs satisfying all active constraints; also reports recovery rate when initial output is invalid.

## What are the main results?

- On **Guess My Number** (15-turn, noisy distance hints that become accurate later): guided agent improves average reward over epochs; baselines (prompting-only, with/without ICL trajectories) do not show consistent improvement.
- On **Wordle** (cumulative hard constraints): adding code/structured generation after an initial phase yields high constraint compliance and better rewards; baselines often violate constraints despite describing them.
- Framing “trustworthiness” as (verifiable rules + enforced generation validity) yields tighter variance / more stable performance.

## How is this similar to GALILEO?

- Same overall thrust: making multi-step agent behavior **auditable** and **reliable** by adding structure around a base LLM.
- Emphasizes explicit **interfaces** and **checks** (validity / constraints) instead of “hopeful” free-form text generation.
- Uses persistent state (“Rule Bank” / procedural memory) learned from trajectories, akin to learned policies/skills rather than pure prompting.

## How is this different from GALILEO?

- Focus is on small, controlled game-like RL environments (Guess-My-Number, Wordle) rather than open-world tool-using agent settings.
- “Reasoning” is mostly extraction of observation→action rules from trajectories; less emphasis (at least in this paper) on rich planning, tool orchestration, or formal proofs.
- Task profiler is an LLM prompt-based classifier that selects strategies, not a learned or formally grounded controller.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO already includes stronger environment modeling, tool semantics, or formalized constraint specs, it can generalize beyond game benchmarks.
- If GALILEO uses more principled verification (e.g., typed actions, programmatic contracts, trace-based checking), it can provide clearer guarantees than “rule bank” heuristics.

## Where GALILEO is weaker / needs to improve

- If GALILEO lacks an explicit *generation-validity gate* (validate, then deterministically recover), this paper suggests a simple, high-leverage reliability mechanism.
- If GALILEO doesn’t track “rule invocation consistency” or compliance ratios, it may miss an important reliability diagnostic beyond task success.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add/describe an explicit **validity-check + fallback** layer for constraint-heavy action spaces (and report recovery rate when fallback triggers).
- [ ] Consider a lightweight **task profiler** (even heuristic) that selects: horizon length, memory usage, and whether to use constrained decoding vs programmatic synthesis.
- [ ] Add metrics analogous to: (a) rule/plan-consistency across steps, (b) constraint-compliance per step (not just end-task success).

## Quotes / details to potentially cite

- Defines trust as: *verifiable* (action-selection reasoning can be inspected/validated) and *reliable* (generated behaviors consistently comply with constraints and feedback).
- “The framework integrates three components: a lightweight task profiler … a reasoning module that learns verifiable observation–action mappings, and a generation module that enforces constraint-compliant outputs through validation or deterministic synthesis.” (Abstract)
