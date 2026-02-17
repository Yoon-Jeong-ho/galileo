# MAD: A Benchmark for Multi-Turn Audio Dialogue Fact-Checking

- Year: 2025
- Venue: SBP-BRiMS 2025 Working Paper (accepted)
- Authors: Chaewan Chun (per arXiv listing)
- URL: https://arxiv.org/abs/2508.12186
- BibTeX key (if we add it): mad2025audio
- Tags: multi-turn, fact-checking, audio, multimodal, dialogue, benchmark

## One-sentence takeaway

MAD introduces a multi-turn *spoken* dialogue fact-checking benchmark with turn/sentence-level labels and audio, showing current models still struggle on multi-turn verification.

## What problem does it solve?

- Existing fact-checking datasets are mostly text-only and/or single-turn; they miss conversational dynamics (multi-turn buildup, challenges, reinforcement) and acoustic complexity (disfluencies, overlap, emotion).
- Need a benchmark where misinformation unfolds across dialogue turns and where systems must (a) detect check-worthy claims and (b) verify claims at sentence and dialogue levels.

## What is the core method / protocol?

- Build **MAD (Multi-turn Audio Dialogues)**: dialogues paired with **audio**.
- Provide rich annotations, including (as described in the abstract):
  - speaker turns
  - dialogue scenarios
  - “information spread styles”
  - sentence-level check-worthiness
  - sentence-level veracity + dialogue-level veracity
- Define two tasks:
  1) check-worthy claim detection
  2) claim verification (sentence-level and dialogue-level)
- Benchmark “strong pretrained models” (not specified in abstract) on these tasks.

## What are the key metrics?

- Claim verification accuracy:
  - sentence level
  - dialogue level
- (Likely standard classification metrics for check-worthiness as well; not specified in abstract.)

## What are the main results?

- Even strong pretrained baselines reach only:
  - **72–74%** accuracy (sentence-level verification)
  - **71–72%** accuracy (dialogue-level verification)
- Authors argue this indicates substantial remaining difficulty, especially for reasoning over speech + dialogue dynamics.

## How is this similar to GALILEO?

- Both are **multi-turn** settings where performance depends on interaction dynamics across turns (not just isolated statements).
- MAD’s **multi-granularity labeling** (sentence vs dialogue) is conceptually adjacent to evaluating “where/when failure happens” in a multi-turn process.
- Emphasizes the need for evaluation protocols that reflect **realistic conversational structure**.

## How is this different from GALILEO?

- Primary focus is **fact-checking/misinformation** in *spoken* dialogues, not pressure-induced belief drift / sycophancy-style instability.
- Includes **audio modality** and acoustic phenomena; GALILEO’s core concerns are typically text-based conversational robustness and belief revision controls.
- Targets check-worthiness + verification, rather than stability/recovery metrics under adversarial social pressure.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO explicitly separates **evidence-driven revision vs drift** and provides controlled multi-turn interventions, that protocol may be cleaner for isolating causal factors than open-ended misinformation dialogues.

## Where GALILEO is weaker / needs to improve

- If GALILEO is text-only, it may miss an important axis of “real-world multi-turn complexity”: **speech/audio** (overlap, disfluency, emotion), which can affect both perception and reasoning.
- MAD’s annotation dimensions (scenario, spread style, check-worthiness) suggest useful “nuisance-factor” axes that GALILEO might not currently cover.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding **multi-level outcome labels** (turn/sentence vs full-dialogue) to better localize failures and enable “time-to-failure” style reporting.
- [ ] Add (even text-only) metadata analogous to MAD’s: **scenario** and **misinfo spread style** (introduce / contest / reinforce) to stratify results.
- [ ] In related work, cite MAD as an example of a **multi-turn benchmark with richer annotation** (and note that audio brings additional uncontrolled complexity).

## Quotes / details to potentially cite

- “We introduce MAD (Multi-turn Audio Dialogues), the first fact-checking dataset aligned with multi-turn spoken dialogues and corresponding audio.”
- “Benchmarking shows that even strong pretrained models reach only 72-74% accuracy at the sentence level and 71-72% at the dialogue level in claim verification…”
