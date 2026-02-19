# AMA: Adaptive Memory via Multi-Agent Collaboration

- Year: 2026
- Venue: arXiv
- Authors: Weiquan Huang; Zixuan Wang; Hehai Lin; Sudong Wang; Bo Xu; Qian Li; Beier Zhu; Linyi Yang; Chengwei Qin
- URL: https://arxiv.org/abs/2601.20352
- BibTeX key (if we add it): ama2026adaptive
- Tags: agent-memory, multi-agent, long-term-consistency, retrieval-routing, conflict-detection, drift-mitigation

## One-sentence takeaway

AMA is a multi-agent external-memory framework that adaptively chooses retrieval granularity (raw/facts/episodes) and explicitly detects+repairs memory conflicts, improving long-context/long-term agent benchmarks while using far fewer tokens than full-context.

## What problem does it solve?

- External memory systems for LLM agents often store and retrieve at a fixed granularity (chunks or coarse summaries), causing either (i) noisy retrieval or (ii) fragmented dependencies that break reasoning.
- Over long interactions, memory stores also accumulate inconsistencies (outdated facts, contradictions), and many systems lack targeted update/delete mechanisms.

## What is the core method / protocol?

- A coordinated 4-role pipeline ("multi-agent" as role specialization):
  - **Constructor**: builds *hierarchical, multi-granular* memory entries from dialogue:
    - Raw Text Memory (utterance + indices/meta)
    - Fact Knowledge Memory (parsed atomic facts; they mention templating via basic S/V/O/C sentence patterns)
    - Episode Memory (triggered summaries when topic shift / explicit request / context window saturation)
  - **Retriever**: rewrites the query to be self-contained and predicts an intent vector to *route* to the right memory granularity (raw vs episode vs facts) and a dynamic top-K.
  - **Judge**: verifies retrieved content (relevance filtering + conflict detection). If evidence is insufficient: retry retrieval (bounded rounds). If contradictions detected: trigger refresh.
  - **Refresher**: performs targeted **update** (default) or **delete** (only for explicit forget requests or retention-limit expiry) to resolve conflicts and keep memory consistent.
- Retrieval is similarity-based over embeddings; they use an external embedding model (text-embedding-3-large) and cosine similarity top-K.

## What are the key metrics?

- LoCoMo benchmark: LLM Score plus F1 and BLEU-1.
- LongMemEvals benchmark: Pass@1 accuracy (LLM-judge-based).
- Efficiency: input tokens and latency (vs FullContext and other memory baselines).

## What are the main results?

- On **LoCoMo** (closed-source backbones), AMA reports higher LLM Score than strong memory baselines; with GPT-4o-mini they report **0.774** vs Nemori **0.740**, and with GPT-4.1-mini **0.805**.
- AMA is reported as the only method surpassing **FullContext** in that setting (FullContext **0.786** when using GPT-4.1-mini in their table text).
- On **LongMemEvals**, AMA reports best average accuracy **0.698**, with large gains on **knowledge-update** tasks (**0.897**).
- **Ablation**: removing the Refresher sharply drops knowledge-update accuracy (they cite **0.897 → 0.568**), supporting the value of explicit conflict resolution.
- **Efficiency**: with default retrieval loop limit Kr=2, they report ~**3613** input tokens vs FullContext **18625** (about 19%), and state "up to ~80%" token reduction.

## How is this similar to GALILEO?

- Targets *multi-turn robustness* under evolving context: keeping the system stable as dialogue length increases.
- Makes drift/instability concrete as an interaction-level failure mode (insufficient evidence, contradictory states) and adds explicit control loops (retry/refresh), which is conceptually aligned with "robustness under pressure" framing.

## How is this different from GALILEO?

- AMA is primarily an **agent memory architecture** paper (storage/retrieval/update), not an evaluation/training method focused on sycophancy/persuasion robustness per se.
- Their control mechanism is *memory lifecycle* (granularity routing + conflict repair), whereas GALILEO (as framed in the rapid-review README) is more about *behavioral robustness across rounds* (belief revision vs drift controls, resisting persuasion/sycophancy).

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO provides targeted stress tests for persuasion/sycophancy and belief revision dynamics, that is more directly tied to alignment failure modes than AMA’s benchmark suite (which is long-term memory QA / long-context tasks).

## Where GALILEO is weaker / needs to improve

- If GALILEO involves extended dialogues, it may need a clearer story for **state persistence** and **conflict repair** (explicitly: how we detect contradictions, decide whether to revise beliefs vs reject user pressure, and how we update stored state).
- AMA provides a concrete template for decomposing responsibilities (retrieval, verification, refresh) that could inspire a more modular GALILEO protocol.

## Action items for GALILEO (experiments / method / writing)

- [ ] In the paper, explicitly separate *retrieval* from *verification* from *update* (even if GALILEO is not a memory system) to clarify "where drift is caught" vs "where drift is prevented".
- [ ] Consider adding a **conflict-detection + repair** ablation for any long-horizon GALILEO setup: e.g., a module that flags contradictions between earlier commitments and current generations, then forces a revise-or-reject decision.
- [ ] If GALILEO evaluates multi-turn robustness, add at least one task class analogous to LongMemEvals' "knowledge update" scenario: user changes a critical fact mid-dialogue and the system must update without becoming sycophantic.
- [ ] Cite AMA for the claim that *adaptive granularity + explicit refresh* materially affects long-horizon performance, and for efficiency-vs-full-context comparisons.

## Quotes / details to potentially cite

- "AMA adopts a multi-agent design that decomposes the memory lifecycle into four functionally distinct yet interdependent roles: the Constructor, Retriever, Judge, and Refresher." (Intro)
- "AMA achieves state-of-the-art performance while reducing token consumption by up to 80% compared to using full context." (Intro / main results summary)
- Refresher ablation: knowledge-update accuracy "0.897" with refresher vs "0.568" without (from their ablation discussion).
