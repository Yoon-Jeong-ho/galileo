# Mem2ActBench: A Benchmark for Evaluating Long-Term Memory Utilization in Task-Oriented Autonomous Agents

- Year: 2026
- Venue: arXiv
- Authors: Yiting Shen, Kun Li, Wei Zhou, Songlin Hu
- URL: https://arxiv.org/abs/2601.19935
- BibTeX key (if we add it): mem2actbench2026shen
- Tags: agents, long-term-memory, benchmark, tool-use, parameter-grounding

## One-sentence takeaway

Mem2ActBench evaluates whether tool-using agents can *proactively apply* long-term memory to produce correctly grounded tool calls from underspecified user requests, and shows current memory frameworks still fail on parameter grounding.

## What problem does it solve?

- Existing “agent memory” benchmarks mostly test *explicit* fact retrieval (user asks a direct question; agent answers), which under-tests realistic assistant settings.
- In persistent assistants, users often issue an underspecified instruction and expect preferences / constraints / state from prior interactions to be implicitly applied (e.g., missing tool arguments).
- Need an evaluation that measures: (1) deciding what to retrieve from long-term memory and (2) *using it to execute* a correct tool invocation (tool choice + argument values).

## What is the core method / protocol?

- Construct long, interruption-heavy multi-session interaction histories that contain scattered “memory facts” about user preferences and task state.
- Build a ground-truth **Fact Evolution Chain** (memory evolution chain) by:
  - Extracting structured facts from dialogue; clustering attributes (BERTopic/HDBSCAN) to group related facts.
  - Resolving local conflicts per attribute and then producing a globally consistent sequence via dependency-graph ordering + cycle-breaking heuristic.
- Generate evaluation tasks via **reverse implicit query generation**:
  - Start from a fully specified gold tool call C=(t*, params) grounded in the memory chain.
  - Generate an underspecified user query that omits key parameter values and relies on references (“use my previous preference”).
  - Filter to ensure **no leakage** and verify memory dependence using a discriminator LLM that must fail without memory.
- Scale/data: merge heterogeneous sources (ToolACE, BFCL, OASST1 noise) into 2,029 sessions; generate 400 tool-use tasks.

## What are the key metrics?

- Primary: correctness of the generated tool invocation (tool selection + parameter values) given (memory, query).
- Human verification rate for “strongly memory-dependent” tasks (reported 91.3%).
- (Paper also reports analyses across multiple memory frameworks; details beyond the intro are not fully captured in this rapid note.)

## What are the main results?

- Human eval: **91.3%** of sampled tasks are strongly memory-dependent (cannot be solved from the final query alone).
- Across **seven** representative memory frameworks, agents remain inadequate at **actively utilizing memory for parameter grounding** in tool calls.

## How is this similar to GALILEO?

- Both care about long-horizon agent behavior where success depends on using accumulated state/knowledge rather than just short-context reasoning.
- Mem2ActBench’s focus on tool invocation correctness aligns with evaluating agentic systems on execution, not just verbal answers.

## How is this different from GALILEO?

- Mem2ActBench is primarily an **evaluation benchmark** for long-term memory application in tool-use tasks; it does not propose a new agent architecture.
- Task format is “given memory + underspecified query → produce tool call”, emphasizing parameter grounding; GALILEO may target broader world-modeling / planning / robustness goals (depending on the paper’s framing).
- Data construction is synthetic/automated from existing tool-use corpora plus conversational noise; may differ from GALILEO’s target environments and training/eval setups.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO includes end-to-end environments or more realistic embodied/web execution loops, it may better capture compounding errors and dynamics beyond single-step tool-call grounding.
- If GALILEO provides principled state representations or learning mechanisms, it could go beyond benchmark-driven diagnosis.

## Where GALILEO is weaker / needs to improve

- If GALILEO does not explicitly evaluate “implicit constraint grounding” from memory into tool parameters, Mem2ActBench suggests a concrete gap to test.
- If GALILEO evaluations rely on explicit questions, it may overestimate memory usefulness compared to underspecified-instruction settings.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding a “memory-to-action parameter grounding” evaluation slice: underspecified user instruction where correct action requires retrieving latent constraints/preferences from prior turns.
- [ ] In related work, contrast **explicit query** memory benchmarks vs. **inference-driven / proactive** memory usage; cite Mem2ActBench as benchmark targeting the latter.
- [ ] When describing GALILEO’s memory usage, be explicit about whether it supports (a) deciding what to retrieve, and (b) applying it to action/tool parameterization.

## Quotes / details to potentially cite

- “Existing benchmarks … primarily test an agent’s ability to passively retrieve isolated facts … They fail to evaluate … actively applying memory to execute tasks.”
- Mem2ActBench simulates “persistent assistant usage” with “long, interrupted interactions” where “preferences and task states” should be “implicitly applied.”
- Scale: “2,029 sessions” and “400 tool-use tasks”; human verification: “91.3% are strongly memory-dependent.”
