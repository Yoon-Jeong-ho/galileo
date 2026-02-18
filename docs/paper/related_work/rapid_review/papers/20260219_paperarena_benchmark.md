# PaperArena: An Evaluation Benchmark for Tool-Augmented Agentic Reasoning on Scientific Literature

- Year: 2025
- Venue: arXiv
- Authors: Daoyu Wang, Mingyue Cheng, Qi Liu, Shuo Yu, Zirui Liu, Ze Guo
- URL: https://arxiv.org/html/2510.10909v1
- BibTeX key (if we add it): paperarena2025wang
- Tags: agents, tool-use, scientific-literature, benchmark, multi-step-reasoning, multimodal

## One-sentence takeaway

PaperArena is a challenging benchmark + evaluation platform (PaperArena-Hub) for measuring tool-augmented agents on realistic cross-paper scientific questions, showing current SOTA agents still perform far below humans and use tools inefficiently.

## What problem does it solve?

- Existing “scientific QA” benchmarks are often single-paragraph / single-paper / tool-free (or solvable without tools), so they under-measure the real workflow of answering research questions that require:
  - cross-document synthesis (multiple papers)
  - multimodal evidence (tables/figures/equations)
  - explicit orchestration of external tools (parsers, retrieval, computation)
- Lack of standardized platform for running/evaluating single-agent and multi-agent tool-using systems on such tasks.

## What is the core method / protocol?

- Build a benchmark from a large corpus of AI papers (open-access sources) and sample a diverse subset using a hierarchical strategy (K-medoids prototypes + farthest-point “boundary” sampling) to reduce topic redundancy.
- Generate candidate QA pairs with a multimodal LLM guided by “tool chain experiences” that reflect realistic research tool use (multi-step, multimodal, cross-paper, cross-database).
- Programmatic verification by executing the tool chain step-by-step to validate intermediate outputs; then human verification and filtering.
- Deliver an evaluation hub (PaperArena-Hub) that:
  - runs agents (single-agent ReAct-like; centralized multi-agent manager+workers)
  - provides a modular tool suite (multimodal parsing, context retrieval, code execution, database querying)
  - logs reasoning traces to analyze tool-use efficiency.

## What are the key metrics?

- Primary: accuracy on question answering (binary correctness via an LLM-as-judge protocol; also compared to human experts on a subset).
- Behavioral: average reasoning steps (tool calls / actions) and a “reasoning efficiency” score based on overlap with a predefined/theoretical tool chain (normalized by executed length).

## What are the main results?

- Benchmark scale: 784 QA pairs (reported as high complexity; evaluation cost comparable to much larger but simpler benchmarks).
- Strong performance gap vs humans:
  - Best reported system (Gemini 2.5 Pro in their multi-agent setup) averages 38.78% accuracy; on the “hard” subset drops to 18.47%.
  - Human PhD-expert baseline reported around 83.5%.
- Tool-use analysis findings:
  - Agents over-invoke tools (inefficient, exploratory use) and show bias toward general-purpose tools (web search, code executor) vs specialized tools.
  - Longer/more complex tool chains correlate with worse performance; failures attributed to both suboptimal planning and flawed tool invocation.
  - If the agent is given the “optimal” tool chain, performance improves, suggesting planning is a major bottleneck (but invocation errors still remain).

## How is this similar to GALILEO?

- If GALILEO targets research assistance (literature understanding, evidence-grounded synthesis), PaperArena is directly relevant as:
  - an evaluation target for tool-using scientific agents
  - a taxonomy of failure modes (planning vs invocation, tool bias, redundancy)
  - a concrete platform pattern (agent lifecycle management: planning/action/memory/reflection)

## How is this different from GALILEO?

- PaperArena is primarily an *evaluation benchmark + hub*, not a new agent method.
- Emphasis is on measuring tool orchestration and cross-paper reasoning via curated QA tasks, rather than optimizing an end-to-end agent for a specific downstream product/workflow.

## Where GALILEO is stronger / cleaner (if true)

- Potential angles (to verify): if GALILEO has stronger planning, tool selection, or trace-level efficiency controls, PaperArena’s analysis offers a clear comparison story.
- If GALILEO emphasizes grounded citation/evidence packaging for writing, that may be beyond PaperArena’s QA-focused scoring.

## Where GALILEO is weaker / needs to improve

- Any GALILEO evaluation that relies mostly on single-document QA or tool-free tasks may not capture the cross-paper + tool orchestration gap PaperArena highlights.
- If GALILEO lacks trace-level metrics (steps/efficiency) or failure-mode attribution, PaperArena suggests useful diagnostics to add.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding PaperArena (or its design principles) as a benchmark reference when motivating the need for tool-augmented scientific agents.
- [ ] If feasible, run GALILEO (or components) on PaperArena-style tasks and report: accuracy, tool-call count, and efficiency vs strong baselines.
- [ ] In GALILEO writing, cite their empirical observation that agents overuse general-purpose tools and degrade with longer chains; position GALILEO as addressing planning/invocation reliability.

## Quotes / details to potentially cite

- Abstract-level headline: “even the most advanced LLM powering a well-established agent system achieves merely 38.78% average accuracy… on the hard subset… 18.47%” (numbers as reported in the paper).
- Key diagnosis: agents “often invoking more tools than necessary” and biased toward general-purpose tools; performance degrades with longer reasoning chains.
- Benchmark framing: tasks require integrating “diverse formats across multiple papers” with “multi-tool orchestration” in realistic research scenarios.
