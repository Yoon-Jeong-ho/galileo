# On the Tool Manipulation Capability of Open-source Large Language Models

- Year: 2023
- Venue: arXiv
- Authors: Qiantong Xu; Fenglu Hong; Bo Li; Changran Hu; Zhengyu Chen; Jian Zhang
- URL: https://arxiv.org/abs/2305.16504
- BibTeX key (if we add it): toolbench_xu_2023
- Tags: tool-use, benchmark, open-source, alignment, prompting

## One-sentence takeaway

ToolBench introduces a diverse tool-manipulation benchmark and shows that relatively lightweight recipes (programmatic data generation + system prompts + in-context demo retrieval + generation-style constraints) can dramatically improve open-source LLM tool-use success, approaching GPT-4 on some tasks.

## What problem does it solve?

- Prior tool-manipulation work often depends on closed APIs; the paper asks whether open-source LLMs can be made competitive for tool-use with practical human effort.
- Identifies recurring failure modes in tool invocation/execution (e.g., formatting, missing required fields, incorrect argument structure, etc.) and targets them with training/prompting “recipes”.

## What is the core method / protocol?

- Benchmark: **ToolBench**, covering “diverse software tools for real-world tasks” (paper reports results over multiple task types).
- Improvement recipe (high-level, from abstract):
  - Train with **usage examples** (alignment via data)
  - Use **system prompts** / **in-context demonstrations**
  - Use a **retriever** for selecting helpful demonstrations
  - Apply **generation style regulation** to reduce tool-calling failures
- Practical supervision claim: on the order of **~1 developer day per tool** to curate/generate the needed data.

## What are the key metrics?

- **Task success rate** on ToolBench (abstract mentions up to **90% success** after applying techniques).
- Comparisons against closed models (abstract claims competitive with **GPT-4** on **4/8** ToolBench tasks).

## What are the main results?

- The proposed enhancement techniques can boost open-source LLMs’ tool manipulation success by **up to ~90%** (as reported in the abstract).
- With the recipe, open-source LLMs can be **competitive with GPT-4 on 4/8 tasks** in ToolBench (abstract).

## How is this similar to GALILEO?

- Shared theme: **robustness under multi-step/multi-turn pressure**—tool use is a concrete setting where small deviations/instabilities compound into failure.
- The “generation style regulation” idea is conceptually adjacent to GALILEO’s interest in controlling drift across turns (reduce unwanted variation that causes downstream failures).

## How is this different from GALILEO?

- This paper focuses on **tool-execution success** (agentic correctness) rather than **social pressure / persuasion / sycophancy** and belief drift vs evidence-based revision.
- Their evaluation target is **API/tool interaction fidelity**, not conversational stability metrics like time-to-flip, recovery, or oscillation patterns.

## Where GALILEO is stronger / cleaner (if true)

- GALILEO can provide **clearer causal controls** for pressure-driven drift vs evidence-driven updating, which ToolBench-style success metrics can conflate.
- GALILEO-style **trajectory metrics** (time-to-failure, recovery-after-failure, oscillations) could diagnose failures more finely than endpoint success.

## Where GALILEO is weaker / needs to improve

- If GALILEO claims to address “robustness in deployed interactive systems,” we may need at least a **concrete agent/task instantiation** (like tool use) demonstrating downstream impact of multi-turn instability.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding a small “**tool-call**” or “structured action” slice where multi-turn pressure can cause formatting/constraint violations; report GALILEO metrics + endpoint success.
- [ ] In writing: cite ToolBench as evidence that **style constraints + demonstrations** can materially affect interactive robustness, supporting the motivation for explicit drift-control mechanisms.

## Quotes / details to potentially cite

- Abstract: introduces ToolBench and claims techniques can “boost leading open-source LLMs by up to **90% success rate**,” becoming “competitive to OpenAI GPT-4 in **4 out of 8** ToolBench tasks,” with about “**one developer day** to curate data for each tool.”
