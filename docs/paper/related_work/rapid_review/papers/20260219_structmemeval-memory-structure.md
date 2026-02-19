# Evaluating Memory Structure in LLM Agents

- Year: 2026
- Venue: arXiv (preprint)
- Authors: Alina Shutova; Alexandra Olenina; Ivan Vinogradov; Anton Sinitsin
- URL: https://arxiv.org/abs/2602.11243
- BibTeX key (if we add it): shutova2026structmemeval
- Tags: agents, long-term-memory, evaluation, benchmark, memory-structure

## One-sentence takeaway

StructMemEval is a benchmark targeting *memory organization/structure* (ledgers, to-do lists, trees, etc.), showing that naive retrieval-style long-term memory struggles while memory agents can do well when explicitly prompted with the right structure.

## What problem does it solve?

- Existing long-term-memory benchmarks for LLM agents largely test *recall* (single-/multi-hop, time updates) and can be handled by relatively simple retrieval-augmented setups.
- As memory architectures become more complex (hierarchies/graphs/OS-like frameworks), there is a need for evaluations that test *structuring knowledge* (not merely retrieving past text).

## What is the core method / protocol?

- Proposes **StructMemEval**, a suite of tasks where the “right” solution involves maintaining a specific memory structure, e.g.:
  - transaction ledgers
  - to-do lists
  - tree-structured information
  - other structured representations (the paper frames these as human-notepad-style organization problems)
- Evaluation compares:
  - simple retrieval-augmented agents (retrieve relevant past text)
  - memory-augmented agents that can *write/update* memory in a structured way
- Key experimental factor: whether the agent is **prompted/hinted** about the appropriate memory structure vs expected to infer it.

## What are the key metrics?

- Task success / correctness on the structured-memory tasks (paper describes “solve tasks reliably” vs “struggle”; exact scoring details are in the benchmark definition).
- Sensitivity to prompting: performance gap between “structure hinted” vs “no hint” conditions.

## What are the main results?

- Simple retrieval-augmented LLMs **struggle** on these structured organization tasks.
- Memory agents can perform **reliably** *if* prompted with how to organize memory.
- Even modern LLMs often **fail to recognize the needed structure** when not explicitly prompted, despite being able to solve abstract algorithmic tasks.

## How is this similar to GALILEO?

- Both are about **evaluation of agent capabilities beyond single-turn QA**, emphasizing skills that emerge over interaction/tooling.
- StructMemEval’s “recognize the right structure without hints” angle resonates with GALILEO’s interest in **multi-turn robustness under pressure**: in both cases, the agent must *infer latent task structure* rather than follow explicit scaffolding.

## How is this different from GALILEO?

- StructMemEval is primarily about **long-term memory organization** (persistent state/knowledge structuring).
- GALILEO (as scoped in this repo) focuses more on **multi-turn robustness failures** (drift, persuasion/sycophancy, belief revision vs stability controls) rather than memory data structures.

## Where GALILEO is stronger / cleaner (if true)

- GALILEO’s framing around **pressure/interaction dynamics** (persuasion, multi-round drift, stability controls) is orthogonal and likely better targeted for those phenomena than a memory-structure benchmark.

## Where GALILEO is weaker / needs to improve

- If GALILEO makes claims about “agent memory” or “statefulness”, it may need **explicit tests for structured state maintenance** (lists/ledgers/trees) to avoid conflating “retrieval” with “organization”.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a small “structured state” slice to GALILEO-style protocols (e.g., multi-round tasks where the agent must maintain a ledger/to-do list while under distractors/pressure) and measure degradation.
- [ ] In the paper related-work, contrast “recall benchmarks” vs “organization benchmarks” and cite StructMemEval as evidence that retrieval-only baselines can be misleading.
- [ ] Consider a condition analogous to their “hint vs no-hint”: evaluate whether models **spontaneously** adopt a stable representation (e.g., a running table/list) vs only when explicitly instructed.

## Quotes / details to potentially cite

- “We propose StructMemEval — a benchmark that tests the agent’s ability to organize its long-term memory, not just factual recall.”
- “Simple retrieval-augmented LLMs struggle with these tasks, whereas memory agents can reliably solve them if prompted how to organize their memory.”
- “Modern LLMs do not always recognize the memory structure when not prompted to do so.”
