# Evaluating Long-Term Memory for Long-Context Question Answering

- Year: 2025
- Venue: arXiv
- Authors: Alessandra Terranova, Björn Ross, Alexandra Birch
- URL: https://arxiv.org/html/2510.23730v1
- BibTeX key (if we add it): Terranova2025EvaluatingLongTermMemoryLoCoMo
- Tags: long-term-memory, long-context, conversational-QA, RAG, episodic-memory, prompt-optimization, agentic-memory, LoCoMo

## One-sentence takeaway

A systematic comparison on LoCoMo shows that memory-augmented QA (RAG / agentic semantic memory / episodic reflections / prompt optimization) can cut tokens by >90% while keeping accuracy competitive, with the “best” memory type depending strongly on the base model.

## What problem does it solve?

- When doing long-context conversational QA, “just stuff the whole dialogue into the prompt” is expensive and still doesn’t produce robust behavior (incl. adversarial unanswerable questions).
- The literature has many memory mechanisms, but fewer *apples-to-apples* evaluations of which memory types help which model classes on a long-dialogue QA benchmark.

## What is the core method / protocol?

- Benchmark: **LoCoMo** (synthetic long-context dialogues; up to ~35 sessions; ~300 turns; ~9k tokens avg), annotated for QA and event summarization.
- QA question types: **single-hop, multi-hop, temporal, open-domain/world knowledge, adversarial** (no answer in dialogue).
- Evaluate multiple “memory architectures” under a unified prompting/eval setup (same prompt regardless of question type):
  - **Full-context prompting** (upper bound baseline).
  - **RAG** semantic memory: retrieve top-*k* utterances (they use **bge-m3** embeddings; main runs use *k*=10) and prepend to prompt.
  - **A-Mem** agentic semantic memory (replicating Xu et al. 2025): structured memory notes with tags/links; retrieve top entries for QA.
  - **PromptOpt** procedural memory: iteratively refine the instruction/prompt using feedback from earlier conversation QA (inspired by LangMem-style prompt optimization).
  - **EpMem** episodic memory: store (q, prediction, label, reflection) from earlier experiences; retrieve a few similar past experiences (top-3) as in-context examples.
  - Combinations: RAG+PromptOpt, RAG+EpMem, RAG+PromptOpt+EpMem.

## What are the key metrics?

- **F1** on QA answers (with instruction to answer using exact words from the dialogue when possible).
- For adversarial questions: score = 1 if the model outputs a “no information available”-style response, else 0.
- **Average token length per query** as an efficiency metric.
- They also report an “average F1 ranking across categories” to compare methods across reasoning types.

## What are the main results?

- Memory-augmented approaches can **reduce token usage by >90%** relative to full-context prompting while staying **competitive in accuracy**.
- The “right” memory complexity depends on model capability:
  - **Smaller foundation models** benefit most from relatively simple **RAG**.
  - **Stronger instruction-tuned / reasoning models** benefit more from **episodic memory (reflections)** and richer/agentic semantic memory.
- Claim/insight: **episodic memory can improve metacognition**, helping models better recognize when information is missing (important for adversarial/unanswerable cases).

## How is this similar to GALILEO?

- Shared concern: *long-horizon coherence* and robust behavior when the conversation/task context is too long or too messy to fit in a single prompt.
- Both emphasize that evaluation should separate **capability** from **systems design choices** (e.g., memory architecture vs raw context length).

## How is this different from GALILEO?

- This is primarily a **memory-architecture bakeoff** on a synthetic long-dialogue QA benchmark (LoCoMo), rather than a targeted protocol for the specific failure modes GALILEO focuses on.
- Their “adversarial” setting is **unanswerable questions** (abstention), not necessarily multi-turn social pressure / drift / recovery dynamics.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO explicitly controls *why* a model changes its answer (evidence vs pressure), that’s a cleaner causal handle than LoCoMo-style QA accuracy alone.
- If GALILEO has explicit multi-turn stability / recovery metrics, that likely covers behaviors not captured by one-shot QA scoring.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently lacks a clear “memory systems” ablation, this paper is a reminder that reviewers may ask for a **token-cost vs performance** comparison against RAG / episodic reflections / prompt-optimization baselines.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a related-work paragraph positioning “memory as a systems intervention” vs “behavioral protocol evaluation”, and cite LoCoMo as an example long-dialogue QA benchmark.
- [ ] Consider a small ablation: **full-context vs RAG vs episodic reflections** for any GALILEO task variant that can be cast as “answer from prior turns”, reporting both **accuracy** and **tokens/query**.
- [ ] If GALILEO includes unanswerable/adversarial queries, consider reporting an explicit **abstention F1** or “correct refusal” metric similar to their adversarial scoring.

## Quotes / details to potentially cite

- "Our findings show that memory-augmented approaches reduce token usage by over 90% while maintaining competitive accuracy." (Abstract)
- "The complexity of the memory architecture should scale with the model’s capabilities: smaller foundation models benefit most from RAG, while more advanced instruction-tuned models [benefit] from episodic memory and richer semantic memory structures." (Intro/Abstract summary)
- LoCoMo description: ten conversations, up to ~35 chat sessions, ~300 turns, ~9,000 tokens average; question types include single-hop/multi-hop/temporal/open-domain/adversarial.
