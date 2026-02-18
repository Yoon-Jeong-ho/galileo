# Beyond a Million Tokens: Benchmarking and Enhancing Long-Term Memory in LLMs

- Year: 2025
- Venue: arXiv
- Authors: Mohammad Tavakoli; Alireza Salemi; Carrie Ye; Mohamed Abdalla; Hamed Zamani; J. Ross Mitchell
- URL: https://arxiv.org/abs/2510.27246
- BibTeX key (if we add it): tavakoli2025beyond
- Tags: memory, long-context, benchmark, multi-turn, retrieval, scratchpad

## One-sentence takeaway

BEAM is a long, coherent conversation memory benchmark (up to ~10M tokens) and LIGHT is a simple cognitive-inspired memory orchestration (episodic retrieval + working window + scratchpad) that boosts long-horizon QA accuracy, highlighting that huge context windows alone still fail as dialogs lengthen.

## What problem does it solve?

- Existing “long-term memory” benchmarks for LLMs often (i) stitch unrelated sessions causing easy segmentation, (ii) focus on narrow personal-life domains, and (iii) mostly test shallow recall rather than broader memory abilities.
- Even with 1M-token contexts, models struggle to reliably use information from very long conversational histories.

## What is the core method / protocol?

- **BEAM benchmark**: automatic pipeline to generate long, narrative-coherent, topically diverse user–assistant conversations (reported up to ~10M tokens) plus probing questions spanning multiple memory abilities.
  - Conversation-plan → recursively refined subplans → chronologically ordered user turns → assistant turns, with follow-ups/clarifications injected for realism.
  - Probing questions target “ten distinct memory dimensions” (paper claim) and are human-validated.
  - Dataset size in abstract: **100 conversations, 2,000 validated questions**.
- **LIGHT framework** (for improving performance): three memory components combined at inference:
  - **Episodic memory**: long-term index over the full conversation + retrieval.
  - **Working memory**: most recent turns.
  - **Scratchpad**: accumulated salient facts written/updated turn-by-turn.

## What are the key metrics?

- Accuracy / performance on BEAM probing questions as a function of dialogue length (long-horizon degradation).
- Improvement vs strong baselines (with/without retrieval augmentation) across multiple backbone LLMs.

## What are the main results?

- Even models with **1M-token context windows** (with or without retrieval augmentation) “struggle as dialogues lengthen” (abstract).
- LIGHT yields average gains of **~3.5% to 12.69%** over the strongest baselines depending on backbone model (abstract).
- Ablations indicate each of the three memory components contributes (abstract).

## How is this similar to GALILEO?

- Shares the high-level motivation: **capabilities can degrade over long multi-turn interactions**, so evaluation should stress *long-horizon* behavior rather than short single-turn snapshots.
- Provides a useful adjacent framing: “large context window” ≠ “robust long-term behavior”; you need explicit mechanisms/controls.

## How is this different from GALILEO?

- Focus is **memory over long-context conversations** (retrieval/scratchpad orchestration), not pressure-driven belief drift / sycophancy / recovery dynamics.
- Primary evaluation seems to be **question answering over a generated dialogue** rather than adversarial or social-pressure interventions.
- Uses synthetic conversation generation; GALILEO likely emphasizes pressure/control conditions (evidence vs pressure) and trajectory metrics (flip/recovery).

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO includes explicit **pressure vs evidence** controls and **recovery-after-failure** objectives/metrics, it provides a clearer causal diagnosis than pure “long conversation QA” accuracy curves.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently lacks extremely-long-horizon settings, BEAM is a reminder that **length scaling can expose new failure modes** even when contexts are technically supported.
- If we don’t already include strong “memory system” baselines, LIGHT-like orchestration could be a missing comparator.

## Action items for GALILEO (experiments / method / writing)

- [ ] Cite BEAM as evidence that **1M-token contexts still degrade** with long dialogs; motivates explicit long-horizon evaluation.
- [ ] Consider adding a baseline inspired by LIGHT: **episodic retrieval + short working window + running scratchpad of salient facts**, then test whether it changes drift/recovery rates under our interventions.
- [ ] If we discuss “memory” at all, clarify whether GALILEO’s failures are *memory retrieval failures* vs *social-pressure preference failures*; BEAM is a contrasting benchmark focused on the former.

## Quotes / details to potentially cite

- “generate long (up to 10M tokens), coherent, and topically diverse conversations… construct BEAM… 100 conversations and 2,000 validated questions.”
- “even LLMs with 1M token context windows (with and without retrieval-augmentation) struggle as dialogues lengthen.”
- “LIGHT… three complementary memory systems: a long-term episodic memory, a short-term working memory, and a scratchpad…”
- “average improvement of 3.5%–12.69% over the strongest baselines.”
