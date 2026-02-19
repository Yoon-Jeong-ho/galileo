# MCPVerse: An Expansive, Real-World Benchmark for Agentic Tool Use

- Year: 2025
- Venue: arXiv
- Authors: Fei Lei et al. (see arXiv for full author list)
- URL: https://arxiv.org/abs/2508.16260
- BibTeX key (if we add it): Lei2025MCPVerse
- Tags: agents, tool-use, benchmark, MCP, evaluation

## One-sentence takeaway

MCPVerse is a large-scale, outcome-based benchmark for LLM agent tool use built on hundreds of *executable* real-world MCP tools, explicitly testing how performance changes as the mounted toolset grows.

## What problem does it solve?

- Existing tool-use benchmarks often (i) rely on synthetic/mock tools, (ii) avoid real execution, and/or (iii) constrain the action space via small mounted tool subsets, making it hard to measure real-world agentic tool competence.
- For time-sensitive information-retrieval tasks, “ground truth” can change; offline static labels can be wrong.

## What is the core method / protocol?

- Tool universe: curated MCP servers/tools collected from MCP hubs (paper claims **550+ executable tools**; and also discusses a larger MCP pool / tool catalog).
- Scale stress-test: present models with increasingly large mounted toolsets (paper describes three modes):
  - **Oracle**: minimal per-question toolset provided.
  - **Standard**: all potentially relevant MCPs provided.
  - **Max-Scale**: very large mounted set (paper frames this as “all MCPs mounted” for maximal action space).
- Evaluation: **outcome-based** scoring rather than strict tool-call matching.
  - For time-invariant tasks: compare final answer to human annotation.
  - For time-sensitive tasks: run scripts to retrieve real-time ground truth.
  - Hybrid judging: LLM-as-judge for semantic consistency + automated checks for environment/state-change tasks.
- Tasks: paper states a benchmark of **~250 tasks**, spanning information retrieval and system operation, with complexity tiers (L1/L2/L3) and time-sensitivity labels.

## What are the key metrics?

- Primary: task **success rate / accuracy** under each mounting mode (Oracle vs Standard vs Max-Scale).
- Secondary: qualitative analysis of failure modes under large toolspaces (e.g., tool selection degradation vs emergent alternative solution paths).

## What are the main results?

- Many models **degrade** as the mounted toolset grows (large action space hurts).
- Some “agentic” models can **benefit from larger toolspaces** (Standard > Oracle), suggesting exploration/unexpected solution paths can offset increased distraction.
- Reported headline: **Claude-4-Sonnet reaches ~44.2% success** in Max-Scale mode (still low, indicating significant headroom).
- Practical limitations are exposed by tool-mount/context constraints (paper notes examples like context-length limits and max-tools-per-request caps for some models).

## How is this similar to GALILEO?

- If GALILEO involves agents operating over tools/APIs (or claims robustness under tool-choice uncertainty), MCPVerse is directly relevant as:
  - a realistic benchmark family to cite,
  - a motivation for evaluating under **large mounted tool sets**, not just oracle retrieval.
- Outcome-based evaluation aligns with systems where multiple tool-call traces can be equally valid.

## How is this different from GALILEO?

- MCPVerse is primarily **benchmark + evaluation harness** work (tool curation, task design, scoring), rather than a new agent architecture.
- It focuses on MCP-style tool ecosystems and “real-world executable tools” rather than simulated environments.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO provides stronger methodological novelty (architecture/learning/objective) or clearer causal ablations, it can complement MCPVerse’s infrastructure contribution.

## Where GALILEO is weaker / needs to improve

- If GALILEO is evaluated only with small retrieved tool subsets (or synthetic tools), it may not address the **scale-of-toolspace** failure mode MCPVerse highlights.
- If GALILEO uses path-matching evaluation (tool name/args correctness), consider shifting toward outcome-based scoring.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a “mounted toolset size” ablation: oracle vs medium vs large toolspace, report degradation curves.
- [ ] Consider outcome-based evaluation (including state-change checks) when multiple traces are valid.
- [ ] Use MCPVerse as related-work evidence that *scale* and *executability* matter for tool-use evaluation.

## Quotes / details to potentially cite

- “MCPVerse integrates more than 550 real-world, executable tools … action space exceeding 147k tokens … outcome-based evaluation with real-time ground truth for time-sensitive tasks.” (from abstract)
- Three-mode setup: Oracle / Standard / Max-Scale (from intro).
