# Exploratory Retrieval-Augmented Planning For Continual Embodied Instruction Following

- Year: 2025 (arXiv); NeurIPS 2024 (journal reference in arXiv)
- Venue: NeurIPS 2024
- Authors: Minjong Yoo, Jinwoo Jang, Wei-jin Park, Honguk Woo
- URL: https://arxiv.org/abs/2509.08222
- BibTeX key (if we add it): yoo2025exrap
- Tags: embodied-agent, continual-instruction-following, LLM-planning, retrieval-augmented, exploration, memory

## One-sentence takeaway

ExRAP augments LLM-based embodied task planning with an explicit, continually-updated environment “context memory” plus information-gain-driven exploration to keep that memory fresh under non-stationarity.

## What problem does it solve?

- Continual / persistent instruction following where *multiple* tasks are active and the environment changes over time (non-stationary).
- Standard LLM planners (SayCan / ProgPrompt / LLM-Planner) repeatedly re-sense / re-query the environment in an ad-hoc way, and do not systematically manage (i) exploration cost vs (ii) memory staleness.

## What is the core method / protocol?

- Maintains an **environmental context memory** that stores time-varying environment facts.
- Each incoming instruction is decomposed into:
  - **queries** over context memory (retrieve relevant environmental state)
  - **task executions** conditioned on query results.
- Introduces **exploration-integrated task planning**:
  - chooses exploration actions during planning using an **information-based exploration** objective (balance memory validity vs exploration load).
- Adds **temporal consistency refinement** for query evaluation to mitigate memory decay / staleness.

(From the paper framing, this is “RAG for embodied planning”, but with an explicit emphasis on dynamic/continual settings and exploration policies that keep the memory synchronized.)

## What are the key metrics?

- Goal success rate (task completion)
- Execution efficiency (e.g., steps / interaction cost; paper claims improvements in efficiency alongside success)
- Robustness across instruction scales/types and degrees of non-stationarity

## What are the main results?

- On VirtualHome, ALFRED, and CARLA continual-instruction settings, ExRAP “consistently outperforms” prior LLM-based planners (ZSP, SayCan, ProgPrompt, LLM-Planner) on success rate and execution efficiency.
- Claimed robustness improvements as non-stationarity increases, attributed to (i) exploration integrated into planning and (ii) temporal-consistency refinement.

## How is this similar to GALILEO?

- Same overall theme: **LLM-based planning for embodied agents** that must ground decisions in environment state.
- Uses an explicit **memory/retrieval** interface between the planner and environment context.
- Highlights the need to manage **interaction cost** (exploration burden) while keeping plans grounded.

## How is this different from GALILEO?

- ExRAP centers on **continual, multi-task instruction following** with explicit **information-gain exploration** to maintain memory validity under non-stationarity.
- Their “retrieval” is primarily from an **agent-built environment memory**, not necessarily from external documents / long-horizon knowledge sources.
- Contribution emphasis is on **when to explore to refresh memory** and **temporal consistency** in query evaluation, rather than (e.g.) GALILEO-style algorithmic structure for planning/learning (depending on what GALILEO’s core method is).

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO already has a clearer abstraction for state representation / belief / uncertainty, it may be cleaner than a heuristic “context memory + refinement” scheme.
- If GALILEO targets broader generalization (beyond maintaining environment facts), it may have a wider scope than ExRAP’s continual-memory synchronization framing.

## Where GALILEO is weaker / needs to improve

- If GALILEO does not explicitly optimize **exploration for information gain** (or equivalently does not treat memory freshness as a first-class objective), ExRAP suggests a concrete gap for continual settings.
- If GALILEO does not address **non-stationarity** (environment changes between/within tasks), ExRAP is a relevant comparison point.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a related-work paragraph framing “continual instruction following in non-stationary environments” and cite ExRAP as exploration-integrated RAG-style planning.
- [ ] Consider adding an ablation/benchmark axis for **non-stationarity** (controlled environment changes) and report success vs interaction cost.
- [ ] If applicable, prototype an **information-gain (or uncertainty)-driven exploration** module whose objective is “keep memory fresh cheaply”, then evaluate against a baseline that re-senses at every step.

## Quotes / details to potentially cite

- “Exploratory Retrieval-Augmented Planning (ExRAP) framework, designed to tackle continual instruction following tasks of embodied agents in dynamic, non-stationary environments.”
- “We implement an exploration-integrated task planning scheme by incorporating the information-based exploration into the LLM-based planning process.”
- “We devise a temporal consistency refinement scheme for query evaluation to address the inherent decay of knowledge in the memory.”
