# Benchmark for Planning and Control with Large Language Model Agents: Blocksworld with Model Context Protocol

- Year: 2025
- Venue: arXiv (submitted to IFAC)
- Authors: Luis Miguel Vieira da Silva; Jayanth Somashekaraiah; Maximilian Weigand; David Kube; Felix Gehlhoff
- URL: https://arxiv.org/abs/2512.03955
- BibTeX key (if we add it): VieiraDaSilva2025BlocksworldMCPBenchmark
- Tags: agents, planning, control, benchmark, blocksworld, tool-use, mcp, partial-observability

## One-sentence takeaway

An MCP-wrapped, executable Blocksworld benchmark with 5 difficulty categories and baseline results for an LLM ReAct agent, designed to evaluate both planning and step-wise execution (incl. constraints + partial observability).

## What problem does it solve?

- Lack of standardized, *executable* benchmarks for comparing LLM agents on planning **and** execution (not just producing a plan string), especially with dynamic interaction and constraint feedback.
- Difficulty comparing agent architectures due to bespoke environment interfaces; they propose MCP as a standard tool API boundary.

## What is the core method / protocol?

- A Blocksworld simulation environment exposed via a REST API, then wrapped as **MCP tools** so any MCP-capable agent can:
  - query rules/state (info tools),
  - verify a complete plan without changing state (verify_plan),
  - execute primitive actions step-by-step (pick_up/put_down/stack/unstack).
- 50 predefined scenarios (JSON) across **five categories**:
  1) Basic,
  2) Requires non-constructive actions,
  3) Impossible/unsatisfiable under constraints,
  4) Additional constraints (block sizes / stacking order),
  5) Partial observability (only top two blocks visible; others unknown).
- Added complexity dimensions beyond classic Blocksworld: limited table positions, block-size constraint (Hanoi-like), and partial observability.

## What are the key metrics?

- Success rate (% solved / correctly classified as impossible).
- Execution time (goal → completion).
- # planning attempts (plan→verify→repair iterations).
- Token consumption (input+output) and derived cost (USD) using a specific model’s pricing.

## What are the main results?

- Baseline single-agent (ReAct in LangGraph; evaluated with OpenAI o3 snapshot o3-2025-04-16) over 50 scenarios (10/category):
  - Category 1: ~80% success; ~76s avg; ~1.1 attempts; ~35.1k tokens.
  - Category 2: ~70% success; ~290s; ~1.7 attempts; ~111.7k tokens.
  - Category 3 (impossible): 100% correct identification; ~125s; ~1.8 attempts; ~18.2k tokens.
  - Category 4: ~70% success; ~732s; ~2.2 attempts; ~143.7k tokens; sometimes misclassifies solvable as unsolvable.
  - Category 5: ~60% success; ~676s; ~3.1 attempts; ~192k tokens; first occurrence of *execution* failures (wrong block-name arguments).
- Takeaway: basic symbolic planning works reasonably, but constraints + partial observability sharply degrade reliability and efficiency.

## How is this similar to GALILEO?

- If GALILEO targets agentic planning/control with tool APIs, this is a close precedent: evaluates an LLM agent that must *plan, verify, execute, recover* against an external tool interface.
- Emphasizes standardized tool interfaces (MCP) for fair comparisons across agent architectures—useful framing for GALILEO evaluation methodology.

## How is this different from GALILEO?

- Domain is symbolic Blocksworld (discrete, toy) rather than real robotic/control/continuous dynamics (if GALILEO is more grounded).
- Uses a fairly “vanilla” single-agent ReAct workflow; no explicit hybrid planner integration or richer memory/state estimation.
- Partial observability is handled by hiding blocks but still within a structured symbolic simulator with crisp constraints and natural-language error messages.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO provides a more realistic environment (continuous control, real sensors, richer failure modes), it can argue higher external validity than Blocksworld.
- If GALILEO provides stronger baselines (hybrid planning, better state tracking, formal guarantees), it can address the paper’s observed failure modes.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently lacks a standardized tool boundary + scenario suite, this paper sets a clear bar: open scenarios, categories, unified metrics, and a protocol interface for agent interchangeability.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adopting an “info / verify / execute” tool taxonomy (or mapping GALILEO tools into these roles) to improve evaluation clarity.
- [ ] Add explicit difficulty categories (constraints, partial observability, impossible cases) and report stratified metrics like they do.
- [ ] Track and report *token/cost* and *# replans* as first-class metrics; they were diagnostic of failures.
- [ ] Include “impossible/unsatisfiable” tasks and score correct detection separately (avoids inflated failure counts).

## Quotes / details to potentially cite

- Benchmark motivation: lack of standardized benchmarks for “planning and execution” of LLM agents (not just static plan generation) and difficulty comparing architectures without a common interface.
- Category design: partial observability = “only the top two blocks of each stack are visible; others unknown”.
- Failure mode: in partial observability, “faulty tool executions occurred for the first time… incorrect block names as argument,” highlighting grounding/tool-call brittleness.
