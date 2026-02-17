# Full-Duplex-Bench: A Benchmark to Evaluate Full-Duplex Spoken Dialogue Models on Turn-taking Capabilities

- Year: 2025
- Venue: ASRU 2025 (per arXiv comment)
- Authors: Guan-Ting Lin; Jiachen Lian; Tingle Li; Qirui Wang; Gopala Anumanchipalli; Alexander H. Liu; Hung-yi Lee
- URL: https://arxiv.org/abs/2503.04721
- BibTeX key (if we add it): lin2025full
- Tags: multi-turn, dialogue, turn-taking, benchmark, speech, full-duplex

## One-sentence takeaway

Full-Duplex-Bench is a scenario-driven, fully-automatic benchmark to diagnose full-duplex spoken dialogue models on timing-centric interaction behaviors (pause handling, backchannels, turn-taking, interruptions) using reproducible metrics.

## What problem does it solve?

- Existing spoken dialogue benchmarks mostly assume half-duplex/turn-based interaction, leaving real-time *timing* behaviors of full-duplex SDMs under-measured.
- Prior work on interaction timing is either coarse (corpus-level gap statistics) or less reproducible (user studies / trained judge tied to a dataset), making cross-model comparisons hard.

## What is the core method / protocol?

- Scenario-driven test set covering four interactive dimensions:
  - Pause handling (don’t “take over” during user hesitations/pauses)
  - Backchanneling (produce brief acknowledgments while user is speaking)
  - Smooth turn-taking (take the turn promptly when appropriate)
  - User interruption management (respond appropriately when user interrupts)
- Evaluation pipeline:
  - Feed a controlled user audio stream into a full-duplex SDM (continuous input, time-synchronous output).
  - Post-process + align the produced output at transcript level.
  - Run an ASR system to obtain word-level timestamps, then compute dimension-specific metrics.
- Simple operational definitions for diagnostic detectors (example: backchannel = <1s and <2 words; “takeover” = any non-silence that is not a backchannel).

## What are the key metrics?

- Behavior detection rates / descriptive statistics per dimension (intended as diagnostic, not prescriptive).
- Explicitly defined “takeover” and “takeover rate (TOR)” for pause-handling-style scenarios.
- Backchannel detection based on duration + word-count thresholds (acknowledged as a simplification).

## What are the main results?

- Primary contribution is the benchmark + automatic evaluation framework (and a taxonomy/table of full-duplex SDMs).
- Paper emphasizes fast, fair, reproducible evaluation for emerging full-duplex SDMs; (specific leaderboard numbers not captured from the truncated HTML extract).

## How is this similar to GALILEO?

- Same overall spirit: *protocol-first evaluation* for multi-turn interaction properties that are not captured by single-turn metrics.
- Focus on diagnosing *trajectory/interaction-level* failure modes rather than only final-answer correctness.

## How is this different from GALILEO?

- Domain is spoken, real-time, full-duplex audio interaction; the measured phenomena are timing/overlap behaviors (pauses, overlaps, interruptions) rather than belief/stance drift or truthfulness under pressure.
- Uses ASR-based alignment + timing heuristics; GALILEO is primarily text-dialogue / content-and-consistency oriented.

## Where GALILEO is stronger / cleaner (if true)

- GALILEO can isolate semantic drift / robustness mechanisms in text with tighter control over content, whereas ASR + timing heuristics introduce measurement noise and threshold sensitivity.

## Where GALILEO is weaker / needs to improve

- GALILEO may under-measure *interaction timing* behaviors that matter in real deployments (especially for voice agents): interruptions, overlap management, and backchannel appropriateness.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider a “timing/turn-taking” appendix section for voice-agent-adjacent robustness: define analogous metrics for *when* the model speaks/acts (interruptibility, overlap tolerance) even in text (e.g., tool calls during user clarification).
- [ ] Borrow the paper’s framing: metrics should be *descriptive diagnostic signals* (not normative) to help developers choose target behavior tradeoffs.

## Quotes / details to potentially cite

- Benchmark focus: “pause handling, backchanneling, turn-taking, and interruption management” (Abstract).
- Backchannel heuristic definition: backchannel segment if (1) duration < 1s and (2) fewer than two words.
- Takeover definition: any non-silence that is not a backchannel; TOR is the average takeover indicator across samples.
