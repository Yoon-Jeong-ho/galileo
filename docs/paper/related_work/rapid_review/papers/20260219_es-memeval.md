# ES-MemEval: Benchmarking Conversational Agents on Personalized Long-Term Emotional Support

- Year: 2026
- Venue: WWW 2026 (The Web Conference)
- Authors: Tiantian Chen, Jiaqi Lu, Ying Shen, Lin Zhang, et al.
- URL: https://arxiv.org/abs/2602.01885
- BibTeX key (if we add it): Chen2026ESMemEval
- Tags: memory, long-term dialogue, personalization, emotional-support, benchmark, dataset, rag

## One-sentence takeaway

A WWW’26 benchmark + dataset for multi-session emotional-support dialogue that probes long-term memory beyond static fact recall (temporal reasoning, conflicts, abstention, user modeling), showing current RAG/long-context LLMs still struggle on evolving user state.

## What problem does it solve?

- Existing “long-term dialogue/memory” benchmarks often reduce to static, explicit fact retrieval, and do not test realistic settings where user info is fragmented, implicit, and changes over time.
- In emotional support scenarios, hallucinating user history or missing evolving context can directly harm personalization and trust; we need targeted evaluation of these capabilities.

## What is the core method / protocol?

- Introduces **ES-MemEval**, a benchmark spanning three task types:
  - Question answering over long, multi-session conversations
  - Summarization of evolving user states
  - Dialogue generation for personalized emotional support
- Defines/evaluates **five core long-term memory capabilities**:
  - Information extraction
  - Temporal reasoning
  - Conflict detection
  - Abstention (knowing when not to answer / when info is missing)
  - User modeling
- Constructs **EvoEmo**, a multi-session dataset for personalized emotional support:
  - 18 virtual users
  - Average ~22.3 sessions and ~23.4 turns per session (per their Table 1)
  - Described as ~510 turns (~13.3k tokens) per user across up to 33 sessions
- Evaluates open-source long-context LLMs, commercial models, and retrieval-augmented (RAG) variants.

## What are the key metrics?

- Not fully extracted in this rapid pass; benchmark covers QA/summarization/generation.
- Practical expectation (based on task types) is a mix of:
  - QA correctness (accuracy / EM / F1 style)
  - Summarization overlap/faithfulness metrics + human/LLM judging
  - Generation quality with a focus on personalization/consistency and reduced hallucination

## What are the main results?

- **Explicit long-term memory (access to histories / stored memories) is critical** to reduce hallucinations and enable personalization.
- **RAG improves factual consistency**, but still **struggles with temporal dynamics** and evolving user state (i.e., retrieving the right time slice and reasoning over changes).
- Personalization correlates strongly with long-term memory; “emotional support” quality is less memory-sensitive and can be produced via general strategies, but risks becoming generic.

## How is this similar to GALILEO?

- Shares the core thesis that **agent evaluation must test long-horizon memory + reasoning**, not just short-context competence.
- The five capability decomposition (IE/TR/conflict/abstain/user modeling) is a useful lens for structuring GALILEO-style long-term evaluation dimensions.
- Highlights failure modes that likely matter for GALILEO too: hallucinated user state/history, temporal inconsistency, and retrieval-time misalignment.

## How is this different from GALILEO?

- Domain focus is **emotional support dialogue**, with “user state evolution” and personalization as central; GALILEO’s target setting may be broader (e.g., general assistants / web agents / tool-using agents).
- The benchmark is built around **dialogue tasks (QA/summarize/generate)** rather than action-taking / tool-use trajectories.
- EvoEmo uses **virtual users** and constructed timelines; depending on GALILEO’s design goals, this may differ from real-user or real-task distributions.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO uses grounded external environments/tools, it may offer clearer, verifiable supervision signals than subjective emotional-support response quality.
- If GALILEO covers broader tool-use or factual task domains, it may generalize evaluation beyond a single vertical.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently emphasizes explicit factual recall, ES-MemEval suggests adding **temporal/conflict/abstention** probes and tests for **implicit, dispersed** info.
- If GALILEO lacks “evolving user model” evaluation, this paper suggests it is a major gap for long-term personalization.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add (or map existing) evaluation axes to: temporal reasoning, conflict detection, abstention, and user modeling; ensure each has targeted test items.
- [ ] Ensure retrieval/memory baselines are evaluated on **time-aware retrieval** (session-level vs turn-level vs summary memory) and measure temporal errors.
- [ ] In related work, cite ES-MemEval as evidence that **RAG ≠ solved long-term memory**, especially for evolving user states.

## Quotes / details to potentially cite

- “ES-MemEval … evaluates five core memory capabilities: information extraction, temporal reasoning, conflict detection, abstention, and user modeling … covering question answering, summarization, and dialogue generation tasks.”
- “EvoEmo … multi-session dataset … capturing fragmented, implicit user disclosures and evolving user states.”
- Table-style stats called out in the paper: **ES-MemEval: 18 conversations/users, avg 22.3 sessions, avg 23.4 turns/session**; EvoEmo described as **~510 turns (~13.3k tokens) across up to 33 sessions per user**.
- “RAG improves factual consistency but struggles with temporal dynamics and evolving user states.”
