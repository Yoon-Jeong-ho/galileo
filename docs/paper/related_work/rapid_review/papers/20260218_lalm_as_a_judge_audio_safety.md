# LALM-as-a-Judge: Benchmarking Large Audio-Language Models for Safety Evaluation in Multi-Turn Spoken Dialogues

- Year: 2026
- Venue: arXiv
- Authors: Amir Ivry; Shinji Watanabe
- URL: https://arxiv.org/abs/2602.04796
- BibTeX key (if we add it): Ivry2026LALMAsAJudge
- Tags: multi-turn, safety, spoken-dialogue, audio, evaluation, LALM, judging

## One-sentence takeaway

A controlled benchmark shows that using large audio-language models as safety “judges” for multi-turn spoken dialogues exposes modality/architecture trade-offs (sensitivity vs stability) and that transcription quality can be a major bottleneck.

## What problem does it solve?

- Safety evaluation for *spoken*, multi-turn voice-agent dialogues is currently text-centric and misses audio cues + is vulnerable to ASR errors.
- Need a controlled benchmark + analysis of whether LALMs can reliably score harmfulness/severity from audio, transcripts, or both.

## What is the core method / protocol?

- Construct **24,000** synthetic English spoken dialogues (3–10 turns).
  - Exactly one turn contains unsafe content.
  - Unsafe content spans **8 harmful categories** and **5 severity grades** (very mild → severe).
- Human validation on **160 dialogues** with **5 raters** to confirm detection reliability + meaningful severity scale.
- Benchmark “judge” models (zero-shot) that output a **scalar safety score in [0,1]** under different input modalities:
  - Audio-only
  - Transcription-only
  - Multimodal (audio + transcript)
- Models: **Qwen2-Audio**, **Audio Flamingo 3**, **MERaLiON**; plus a transcription-only **LLaMA** baseline.
- Evaluate:
  - **Sensitivity** (detect unsafe content)
  - **Specificity for ordering severity levels** (does score increase with grade?)
  - **Stability across dialogue turns** (score consistency as conversation progresses)

## What are the key metrics?

- Sensitivity to unsafe detection (true positive behavior).
- Severity ordering quality (monotonicity / ranking across 5 grades).
- Turn-level stability of safety score.
- (Analysis dimension) impact of transcription quality (e.g., Whisper-Large) on judge behavior.

## What are the main results?

- Strong **trade-offs by architecture and modality**:
  - Most sensitive judge configuration can be **least stable across turns**.
  - More stable configurations may **miss mild harmful content**.
- **Transcription quality is a key bottleneck**:
  - Using **Whisper-Large** transcripts may **significantly reduce sensitivity** in transcription-only modes.
  - Yet severity ordering may be **largely preserved** even when sensitivity drops.
- **Audio helps** when paralinguistic cues matter or when transcription fidelity is category-critical.

## How is this similar to GALILEO?

- Focus on evaluation protocols/benchmarks for model behavior in safety-relevant settings.
- Emphasizes *multi-turn* dynamics and measurement beyond single-turn classification.

## How is this different from GALILEO?

- Specific to **spoken dialogue safety judging** with **audio-language models**.
- Outputs a continuous safety score and studies modality effects (audio vs transcript).

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO targets more general or more realistic deployment settings, it may provide broader external validity than synthetic controlled dialogues.

## Where GALILEO is weaker / needs to improve

- If GALILEO relies on transcript-only evaluation, it may miss audio-specific harms and ASR-induced failure modes highlighted here.
- Need to explicitly account for *turn-level stability* as a metric if not already.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add/borrow metrics for **turn-level stability** of safety judgments in multi-turn interactions.
- [ ] If relevant, include an ablation on **ASR quality** (clean vs noisy transcripts; different ASR models) to quantify sensitivity drops.
- [ ] Consider a controlled synthetic benchmark slice (single unsafe turn; graded severity) to stress-test ordering and detection in a reproducible way.
- [ ] In writing, call out that transcript-only safety evaluation can be systematically biased by ASR errors.

## Quotes / details to potentially cite

- “We present LALM-as-a-Judge, the first controlled benchmark and systematic study of large audio-language models (LALMs) as safety judges for multi-turn spoken dialogues.”
- “We generate 24,000 unsafe and synthetic spoken dialogues … with one of 8 harmful categories … and … 5 grades, from very mild to severe.”
- “Whisper-Large may significantly reduce sensitivity for transcription-only modes, while largely preserving severity ordering.”
