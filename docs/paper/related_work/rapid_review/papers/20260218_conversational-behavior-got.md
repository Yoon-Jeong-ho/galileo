# Conversational Behavior Modeling Foundation Model With Multi-Level Perception

- Year: 2026
- Venue: arXiv (authors mention ICML in the HTML version)
- Authors: Dingkun Zhou; Shuchang Pan; Jiachen Lian; Siddharth Banerjee; Sarika Pasumarthy; Dhruv Hebbar; Siddhant Patel; Zeyi Austin Li; Kan Jen Cheng; Sanay Bordia; Krish Patel; Akshaj Gupta; Tingle Li; Gopala Anumanchipalli
- URL: https://arxiv.org/abs/2602.11065
- BibTeX key (if we add it): zhou2026conversational
- Tags: conversation-modeling, speech-acts, full-duplex, streaming, rationales, graph-of-thoughts, synthetic-data

## One-sentence takeaway

A streaming, strictly-causal full-duplex conversation framework that (i) predicts hierarchical speech acts at 1 Hz and (ii) generates low-latency, auditable rationales via a Graph-of-Thoughts trained from teacher reasoning traces.

## What problem does it solve?

- Full-duplex spoken dialogue requires per-second (or faster) decisions about conversational behavior (turn-taking, backchannels, interruptions, etc.) under hard latency constraints.
- Prior duplex modeling often frames the task as sequence prediction (next segment / next dual-token), which can bypass an explicit intent-to-action reasoning layer and produce opaque decisions.
- They want an interpretable, causal, real-time "perception -> reasoning -> generation" loop for conversational behavior.

## What is the core method / protocol?

- **Multi-level perception (hierarchical speech act perceiver):**
  - Predict **high-level** speech-act intent categories (e.g., constative / directive / commissive / acknowledgment).
  - Condition **low-level** interaction behaviors (e.g., backchannel / interruption / turn-taking / continuation) on the high-level state.
  - Operates in a **strictly causal, streaming** setting with 1-second blocks.
  - Architecture sketch from the HTML:
    - Fuse frozen acoustic + semantic embeddings with a learned gate.
    - Two causal transformer decoders (high/low streams), with FiLM-style modulation of low-level stream using high-level state.

- **Graph-of-Thoughts (GoT) rationale generation:**
  - Uses teacher LLM reasoning traces during data construction as *supervision*, not as a runtime reasoning engine.
  - At inference, builds a sliding-window graph with **second-level nodes** and **committed sentence nodes**.
  - Stage-1 selects an evidence/topic chain (anchors) from prior sentence nodes; stage-2 generates a short rationale conditioned on selected evidence + local caches.
  - Designed to control latency (their comparison table reports sub-second rationale latency for GoT vs multi-second+ for direct LLM prompting, and much slower for a stronger thinking model).

- **Dataset: ConversationGoT-120h:**
  - ~120 hours of synthetic open-domain chit-chat dialogues with controllable, "event-rich" interaction patterns.
  - Per-second hierarchical labels + rationale annotations generated under causal constraints, then human-verified.
  - Synthetic audio via TTS with diverse voices.

## What are the key metrics?

- For speech-act perception:
  - Per-class **F1** and **AUC** (noting label imbalance at low level).
  - In-domain evaluation on ConversationGoT-120h.
  - OOD transfer evaluation to a real dataset (Candor).
  - Human-model agreement rates for per-second decisions.

- For rationale generation:
  - Subjective rubric-style scoring by an automatic judge (GPT-4o in their setup) across dimensions like alignment/justification/completeness/clarity.
  - **Latency** comparisons across methods (GoT vs random anchors vs directly prompting LLMs).

## What are the main results?

- The hierarchical perceiver shows strong discrimination on common low-level behaviors (they report AUCs around ~0.94-0.97 for turn-taking/continuation in-domain), with lower F1 for long-tail categories (interruption/backchannel) but still decent AUC.
- OOD transfer to Candor degrades only slightly (per their claim), suggesting synthetic data/labels transfer to real conversations.
- GoT rationales improve quality vs random evidence selection, and deliver much lower runtime latency than directly prompting an LLM, while approaching stronger (but slower) LLM rationale quality.

## How is this similar to GALILEO?

- Treats interaction as **structured behavior** rather than raw token prediction, and emphasizes **interpretable intermediate structure** (hierarchical labels + explicit evidence chains).
- Positions the work as a **benchmarking / evaluation** foundation: label taxonomy + dataset + metrics (and an interpretable reasoning artifact) rather than only end-task generation.
- Highlights **causality / leakage control** in data construction, which is often a key concern in agent evaluation settings.

## How is this different from GALILEO?

- Focus is **spoken, full-duplex, real-time** interaction with per-second outputs; likely more low-level timing/overlap centric than GALILEO's typical textual agent settings.
- Uses a specific hierarchical speech-act taxonomy and builds a domain-specific synthetic corpus + TTS audio pipeline.
- Rationale generation is framed as *behavior explanation* (per-second) rather than long-horizon planning or tool-use reasoning.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO targets broader agentic reasoning/evaluation beyond speech, it may provide more general task families and less domain-specific assumptions than a duplex speech-act pipeline.
- GALILEO may not require synthetic-audio generation or specialized streaming ASR/speaker-tracking components.

## Where GALILEO is weaker / needs to improve

- If GALILEO lacks a real-time/streaming setting, this paper is a reminder that **latency-bounded, causal** evaluation protocols matter for interactive agents.
- If GALILEO does not include explanation-latency tradeoffs, this paper provides a concrete template for reporting **quality vs runtime** for interpretability.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding a "strictly causal / streaming" evaluation variant (even for text) where models only see a bounded past window and must act at fixed ticks.
- [ ] Consider hierarchical labeling where a coarse "intent" label conditions finer-grained behavioral predictions; evaluate whether hierarchy improves stability on long-tail behaviors.
- [ ] In related work: cite this as an example of **training explanation models from teacher traces** to get interpretable outputs with controlled latency.
- [ ] If we do rationales: report **latency** and **auditability** explicitly (not just quality), mirroring their GoT vs LLM baselines framing.

## Quotes / details to potentially cite

- "Our system performs online prediction at the per-second level" and is "strictly causal and streaming" (Method section).
- ConversationGoT-120h: a ~120h dataset with per-second two-level speech acts and human-verified rationales.
- Their core framing: shift from "next-token prediction" toward "next-behavior reasoning" for full-duplex interaction.
