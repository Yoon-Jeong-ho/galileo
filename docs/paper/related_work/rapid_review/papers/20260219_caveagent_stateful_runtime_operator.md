# CaveAgent: Transforming LLMs into Stateful Runtime Operators

- Year: 2026
- Venue: arXiv
- Authors: Maohao Ran, Zhenglin Wan, Cooper Lin, Yanting Zhang, Hongyu Xin, Hongwei Fan, Yibo Xu, Beier Luo, Yaxin Zhou, Wangbo Zhao, Lijie Yang, Lang Feng, Fuchao Yang, Jingxuan Wu, Yiqiao Huang, Chendong Ma, Dailing Jiang, Jianbo Deng, Sirui Han, Yang You, Bo An, Yike Guo, Jun Song
- URL: https://arxiv.org/abs/2601.01569
- BibTeX key (if we add it): ran2026caveagent
- Tags: agents, tool-use, state, runtime, code-generation, long-horizon

## One-sentence takeaway

Make the *Python runtime* (persistent objects) the primary state for an LLM agent, using the LLM mostly as an orchestrator that writes code to manipulate/inspect that runtime, which reduces context drift and token costs on long-horizon tasks.

## What problem does it solve?

- Standard tool-using agents are largely *text/JSON centric*: each tool call is a stateless transaction, intermediate results must be serialized back into context, and multi-step control flow becomes fragile across many turns.
- This leads to long-horizon brittleness (context drift), high token overhead (printing/rehydrating data structures), and limited composability (loops/conditionals spread across turns).

## What is the core method / protocol?

- **Dual-stream architecture**:
  - *Semantic stream*: lightweight natural-language reasoning/orchestration.
  - *Runtime stream*: a **persistent Python runtime** treated as the central workspace.
- **Stateful Runtime Management**:
  - Inject / manipulate / retrieve complex Python objects that persist across turns (e.g., DataFrames, DB connections), avoiding repeated textual serialization.
  - Treat “code as action, state as memory”: store intermediate artifacts as variables and refer to them later.
- **Runtime-integrated skill management**:
  - Extends an “Agent Skills” open standard with executable skill injections (interoperability via runnable skills rather than only schema-level descriptions).
- Additional framing:
  - Runtime state is programmatically inspectable → can support automated evaluation and reward signals (they position this as enabling RL with verifiable rewards).

## What are the key metrics?

- Success rate on benchmark tasks (explicitly mentions **Tau2-bench**; also **BFCL**).
- Token consumption / efficiency for multi-turn scenarios.
- Qualitative case studies (e.g., data-intensive tasks; multi-agent coordination via shared runtime state).

## What are the main results?

From the paper’s abstract (v2):

- **+10.5%** success-rate gain on retail tasks (Tau2-bench context).
- **-28.4%** total token consumption for multi-turn scenarios.
- **-59%** token consumption on data-intensive tasks.
- Handles data scales that overflow context windows in JSON-based and “code-based but text-bound” agents.

## How is this similar to GALILEO?

- Shares the motivation of **robust long-horizon agent behavior** and reducing dependence on fragile text-only context.
- Emphasizes **structured state** and **verifiability/inspectability** as a way to make agent behavior easier to evaluate and debug.

## How is this different from GALILEO?

- CaveAgent’s central claim is architectural: **persistent Python runtime as the primary locus of state**; many systems (including code-generating agents) still “textualize” state back into prompts.
- Heavy emphasis on **object injection / retrieval** (native objects as interfaces), rather than only tool-call traces + textual summaries.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO already formalizes state transitions / memory with explicit schemas and invariants, it may offer clearer **cross-language** or **platform-agnostic** guarantees than “Python runtime as SSOT.”

## Where GALILEO is weaker / needs to improve

- If GALILEO relies on summarization or text logs as the main memory carrier, CaveAgent is a concrete argument that **native runtime objects** can be a more stable memory substrate (and cheaper in tokens).

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider a “runtime-as-memory” baseline in related work: persistent interpreter state with object references, compared to prompt-only / JSON tool-calling.
- [ ] Add a discussion point: **textualization bottleneck** (need to serialize objects into text to persist/share) and how GALILEO avoids or embraces it.
- [ ] If applicable, evaluate token/latency tradeoffs for multi-turn tasks when state is held in a runtime vs. reprinted/reparsed each turn.

## Quotes / details to potentially cite

- “Shift[s] tool use from ‘LLM-as-Text-Generator’ to ‘LLM-as-Runtime-Operator.’”
- “CaveAgent elevates the persistent Python runtime as the central locus of state, with a lightweight semantic stream serving as its orchestrator.”
- Reported improvements: “10.5% success rate gain… 28.4% reduction in total token consumption… 59% token reduction on data-intensive tasks.”
