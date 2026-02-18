# MCPAgentBench: A Real-world Task Benchmark for Evaluating LLM Agent MCP Tool Use

- Year: 2025 (arXiv v1 Dec 31, 2025; v3 Jan 21, 2026)
- Venue: arXiv
- Authors: Zixiang Liu; Elsie Dai; Wenhan Yu; Lei Yu; Tong Yang; Jinjun Han; Hong Gao
- URL: https://arxiv.org/abs/2512.24565
- BibTeX key (if we add it): mcpagentbench2025
- Tags: agents, tool-use, MCP, benchmark, evaluation, efficiency-metrics

## One-sentence takeaway

MCPAgentBench is an Autogen-based, locally runnable benchmark for MCP tool-use that stresses tool selection under distractors and evaluates not only correctness but also *efficiency* (order/parallelism, time, tokens).

## What problem does it solve?

- Existing MCP tool-use benchmarks often depend on live/remote MCP servers (hurting stability/reproducibility) and focus mainly on task correctness/protocol compliance.
- They provide limited, coarse difficulty characterization of tool-invocation complexity.
- They generally lack explicit metrics for execution efficiency (time/tokens, and whether the agent used the minimal/efficient invocation plan).

## What is the core method / protocol?

- Collects “authentic” MCP tool definitions from public MCP directories/marketplaces and reconstructs them as *local* simulated MCP servers/tools.
- Builds task instances by curating real-world task descriptions and manually aligning each task to a *unique* tool-invocation solution.
- Evaluates agents in a dynamic sandbox (built on Autogen) where each task provides a candidate tool list containing:
  - the correct tool(s) and
  - many “distractor” tools that are unrelated or confusable,
  to test tool discrimination/robust selection.
- Classifies tasks by invocation complexity:
  - single-tool
  - dual-tool parallel
  - dual-tool serial
  - multi-tool (mix of serial/parallel)
  and by domain (daily vs professional).

## What are the key metrics?

- Task Finish Score (TFS): task is finished iff the *set* of tool invocations matches the gold solution (tool names + parameters as applicable), ignoring order.
- Task Efficiency Finish Score (TEFS): task is efficiently finished iff it is finished *and* the invocation sequence matches gold serial/parallel structure (i.e., order/parallelism correctness).
- Resource efficiency:
  - Time efficiency: TEFS-weighted score per minute.
  - Token efficiency: TEFS-weighted score per 1k output tokens.

## What are the main results?

- Constructs from raw sources (after dedup / curation):
  - ~9,714 MCP servers and 20,000+ MCP tools (definitions), and
  - 178 human-curated, high-quality task instances drawn from an initial pool of 841 tasks.
- Reports “significant performance differences” among mainstream LLMs, especially on complex multi-step tool invocations, and motivates efficiency-aware evaluation beyond pass/fail correctness.

## How is this similar to GALILEO?

- Same broad goal: evaluate agentic tool-use/planning/execution on realistic tasks and tools.
- Emphasizes structured difficulty (single-step vs multi-step; serial vs parallel), which aligns with GALILEO’s need to stratify “agent difficulty”.
- Uses an execution-grounded setup (simulated tools with schema/parameters) rather than pure text-only evaluation.

## How is this different from GALILEO?

- MCPAgentBench is specifically scoped to MCP tool invocation and Autogen-based sandboxing, with evaluation defined around matching a gold tool-invocation plan.
- The benchmark’s “needle-in-haystack” candidate list with distractor tools is central; GALILEO may (depending on its design) target broader environments, modalities, or richer interaction loops.
- TFS ignores invocation order (set match), while TEFS enforces serial/parallel structure; GALILEO’s scoring may differ (e.g., outcome-based success, partial credit, safety constraints, etc.).

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO evaluates end-to-end outcomes (not just matching a predefined invocation trace), it may be more robust to alternative correct strategies.
- If GALILEO includes richer environment dynamics, longer horizons, or non-MCP tool interfaces, it may provide broader external validity.

## Where GALILEO is weaker / needs to improve

- Consider adding explicit efficiency-aware metrics analogous to TEFS/time/token efficiency, not just success rate.
- Consider adding “distractor tool lists” or confusable tools to stress tool selection rather than only parameter filling.
- Ensure reproducibility by preferring local/sandboxable tool backends where possible.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add an “efficient plan match” metric: distinguish (a) task success from (b) success with minimal/optimal call graph (serial/parallel correctness), similar to TEFS.
- [ ] Add time/token efficiency reporting as first-class metrics (score per minute; score per 1k tokens).
- [ ] Introduce tool candidate lists with distractors to measure tool discrimination and robustness.
- [ ] In related work, position GALILEO relative to MCP benchmarks by highlighting (i) reproducibility approach (local vs remote), (ii) evaluation target (trace-match vs outcome), and (iii) efficiency measurement.

## Quotes / details to potentially cite

- “We collect authentic 841 tasks and over 20000 MCP Tools… After deduplication, we obtain definitions for 9714 MCP servers…” (data scale).
- “The evaluation employs a dynamic sandbox environment that presents agents with candidate tool lists containing distractors…” (needle-in-haystack tool selection).
- Metrics: Task Finish Score (TFS) vs Task Efficiency Finish Score (TEFS), plus time/token efficiency.
