# Full-Duplex-Bench: A Benchmark to Evaluate Full-duplex Spoken Dialogue Models on Turn-taking Capabilities

- Year: 2025
- Venue: ASRU 2025 (arXiv)
- Authors: Guan-Ting Lin; Jiachen Lian; Tingle Li; Qirui Wang; Gopala Anumanchipalli; Alexander H. Liu; Hung-yi Lee
- URL: https://arxiv.org/abs/2503.04721
- BibTeX key (if we add it): fullduplexbench2025lin
- Tags: multi-turn, spoken-dialogue, full-duplex, turn-taking, benchmark, evaluation

## One-sentence takeaway

Full-Duplex-Bench is a scenario-driven, fully automatic benchmark to diagnose full-duplex spoken dialogue models’ real-time interaction behaviors (pause handling, backchanneling, turn-taking, interruptions).

## What problem does it solve?

- Full-duplex spoken dialogue models (SDMs) can *listen and speak simultaneously*, but evaluation has lagged: most benchmarks assume turn-based (half-duplex) interactions or rely on coarse corpus-level timing stats / non-reproducible user studies.
- Need a fair, fast, reproducible evaluation protocol that focuses on *interaction timing behaviors* rather than only content quality.

## What is the core method / protocol?

- Define scenario-driven test cases that elicit four key behaviors:
  - pause handling
  - backchanneling
  - turn-taking
  - user interruption management
- Run a full-duplex SDM on user audio streams to get time-synchronous outputs.
- Post-process to align user/model streams at the transcript level and apply automatic detectors/metrics per behavior.
- Emphasis: metrics are intended to be *descriptive diagnostics* (not necessarily a single scalar objective), supporting rapid iteration and cross-model comparison.

## What are the key metrics?

- Automatic behavior-detection metrics for:
  - whether/when the model backchannels
  - whether/when the model yields/claims the floor (turn-taking)
  - whether the model properly handles pauses
  - interruption handling (e.g., stopping/adjusting when the user speaks)

(Details are in the benchmark definitions; the paper positions them as objective and reproducible across models.)

## What are the main results?

- Primary contribution is the benchmark + toolkit release rather than a new model; the paper argues the framework enables consistent, rapid evaluation of full-duplex interactive behaviors across many SDMs.
- Provides an overview of contemporary full-duplex SDMs and highlights missing/opaque capabilities in many public systems.

## How is this similar to GALILEO?

- Shared theme: *multi-turn interaction evaluation* with diagnostics that go beyond single-turn/task accuracy.
- Similar framing: evaluation should be reproducible, scenario-driven, and should isolate specific failure modes/behaviors.

## How is this different from GALILEO?

- Targets *spoken*, real-time, full-duplex interaction (timing, overlaps, interruptions), not textual belief/stance robustness.
- Focuses on interaction *dynamics* and paralinguistic/turn-taking behaviors rather than semantic robustness to persuasion/misinformation.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO is about textual multi-turn robustness (drift, persuasion, belief revision), it can more directly connect behaviors to truthfulness/consistency objectives and controlled perturbations.

## Where GALILEO is weaker / needs to improve

- If GALILEO ignores real-time interactive dynamics (timing/backchannels/interruptions), it may miss an important class of “multi-turn” failures that appear in voice agents and could correlate with user influence/pressure.

## Action items for GALILEO (experiments / method / writing)

- [ ] In related work: cite as an example of *scenario-driven, diagnostic multi-turn evaluation*—a parallel to how GALILEO decomposes multi-turn robustness into measurable behaviors.
- [ ] Consider whether GALILEO should explicitly declare scope as text-only, and (optionally) mention that spoken full-duplex evaluation adds another axis of multi-turn robustness.

## Quotes / details to potentially cite

- Benchmark evaluates “pause handling, backchanneling, turn-taking, and interruption management” with “automatic metrics” for “fair, fast evaluation” (from abstract).
- “The metrics are intentionally descriptive rather than prescriptive, allowing developers to prioritize behaviors according to specific application requirements.” (from introduction)
