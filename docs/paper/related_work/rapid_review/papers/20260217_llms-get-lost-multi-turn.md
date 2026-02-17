# LLMs Get Lost In Multi-Turn Conversation

- Year: 2025
- Venue: arXiv
- Authors: Philippe Laban; Hiroaki Hayashi; Yingbo Zhou; Jennifer Neville
- URL: https://arxiv.org/abs/2505.06120
- BibTeX key (if we add it): Laban2025LostInConversation
- Tags: multi-turn, underspecification, reliability, evaluation, simulation, recovery-adjacent

## One-sentence takeaway

Across six generation tasks and 15 model families, multi-turn *underspecified* conversations cause a large performance drop that is driven mostly by skyrocketing **unreliability** (best vs worst run gap), not by a big loss of best-case capability.

## What problem does it solve?

- Most LLM eval is single-turn and fully specified, but real use often starts underspecified and gets clarified over multiple turns.
- Prior multi-turn benchmarks are often “episodic” (turns are separable subtasks), which can overestimate real conversational robustness.
- Need a controlled, scalable way to compare *the same underlying tasks* in single-turn vs multi-turn settings and quantify *how* things break.

## What is the core method / protocol?

- **Sharded instruction construction**: take a fully-specified instruction from an existing benchmark and split it into “shards”:
  - Shard 1 expresses high-level intent; later shards progressively reveal constraints/details.
- **Sharded conversation simulation** (multi-turn underspecification):
  - A user simulator reveals at most one shard per turn (rephrased to fit the conversation), chosen based on conversation state.
  - Assistant model responds freely each turn.
  - A system component classifies assistant responses into strategies (e.g., clarification, hedging, answer attempt, etc.).
  - When the assistant makes an **answer attempt**, an answer extractor pulls the evaluable span (e.g., code, SQL, number), which is scored by a task-specific evaluator.
  - Conversation ends when the evaluator marks an answer correct or when shards run out.
- Additional simulation types to isolate factors:
  - **Full**: original single-turn instruction (baseline).
  - **Concat**: all shards concatenated in one turn (controls for sharding rephrasing/information loss).
  - **Recap**: sharded convo + final turn that recaps all shards (a simple intervention).
  - **Snowball**: each new turn repeats all previously revealed shards + the new shard (turn-level recap).
- Scale: ~600 instructions (90–120 per task), **15 LLMs**, **N=10** stochastic runs per (model, instruction, condition) → **200k+** simulated conversations.

## What are the key metrics?

- Task scores mapped to a common **0–100** scale.
  - Code / Database (text-to-SQL) / Actions (function calling) / Math: binary correctness mapped to {0, 100}.
  - Data-to-text: **BLEU**.
  - Summary (Summary-of-a-Haystack): an LLM-judge **Joint Score** focused on coverage + attribution accuracy.
- From repeated runs per instruction (scores S):
  - **Average performance**: mean(S)
  - **Aptitude (A^90)**: 90th percentile of S (best-case proxy)
  - **Unreliability (U^90_10)**: percentile_90(S) − percentile_10(S) (best–worst gap); “reliability” sometimes reported as 100 − U.

## What are the main results?

- Universal degradation in underspecified multi-turn:
  - Average drop from single-turn Full to multi-turn Sharded is ~**39%** (avg across 6 tasks).
  - Concat is close to Full (~95% of Full on average), suggesting the big drop is *not* from information loss in sharding.
- Decomposition:
  - Aptitude drops modestly (reported ~**16%** avg), while **unreliability more than doubles** (reported ~**+112%** avg).
  - Intuition: models often can do the task *sometimes* in multi-turn, but are much less consistent; when they “take a wrong turn,” they often fail to recover.
- Observed failure dynamics (qualitative):
  - Early incorrect assumptions; premature “final solution” attempts; over-reliance on previous (wrong) attempts; verbosity correlates with more assumptions.
- Reasoning/test-time-compute does not automatically fix this:
  - Reasoning models included (e.g., o3, DeepSeek-R1) degrade similarly; longer responses can worsen assumption accumulation.

## How is this similar to GALILEO?

- Central shared theme: **multi-turn robustness as a trajectory property**, not a single-turn score.
- Provides a clean precedent for separating:
  - best-case capability (aptitude) vs
  - *stochastic* instability / “time-to-get-lost” style behavior (unreliability).
- Their “lost in conversation” phenomenon is conceptually adjacent to GALILEO’s focus on **drift/flip and (non-)recovery** under extended interaction.

## How is this different from GALILEO?

- Pressure source differs:
  - This paper studies **underspecification / gradual revelation**, not explicit **social pressure / persuasion / authority**.
- Outcome differs:
  - Focus is aggregate success and reliability ranges, not belief revision vs drift controls, nor explicit “recovery-after-flip” measurements.
- Uses an LLM-based simulator + classifier/extractor pipeline; GALILEO may prefer protocols with fewer learned components (or at least explicit ablations).

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO has explicit **pressure-vs-evidence controls** and **recovery trajectories**, that is more diagnostic than underspecification-only degradation.
- GALILEO can frame robustness failures as *social influence susceptibility* rather than generic conversational brittleness.

## Where GALILEO is weaker / needs to improve

- This paper’s metrics explicitly separate **aptitude vs unreliability** via repeated-run percentiles; if GALILEO reports only mean/flip-rate, it may miss the “best-case vs worst-case” story.
- Their “Concat” control is a strong method to argue the failure is due to multi-turn/underspecification, not rephrasing; GALILEO should have analogous controls (e.g., pressure-free paraphrase controls).

## Action items for GALILEO (experiments / method / writing)

- [ ] Add an “aptitude vs unreliability” reporting layer (e.g., percentile-based best-case and best–worst gap) for multi-turn protocols.
- [ ] Consider a simple “recap” intervention baseline (end-of-dialogue recap) and a “snowball” baseline (repeat context each turn) to test whether failures are memory/recall vs genuine drift.
- [ ] In writing, cite this as evidence that **multi-turn interaction reliability is a distinct failure mode** even when single-turn performance is strong.

## Quotes / details to potentially cite

- “...all the top open- and closed-weight LLMs we test exhibit significantly lower performance in multi-turn conversations than single-turn...” (Abstract)
- “...decomposes the performance degradation into two components: a minor loss in aptitude and a significant increase in unreliability.” (Abstract)
- Metric definitions:
  - Aptitude A^90 = percentile_90(S)
  - Unreliability U^90_10 = percentile_90(S) − percentile_10(S)
