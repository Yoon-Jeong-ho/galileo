# Audio MultiChallenge: A Multi-Turn Evaluation of Spoken Dialogue Systems on Natural Human Interaction

- Year: 2025
- Venue: arXiv
- Authors: Advait Gosai, Tyler Vuong, Utkarsh Tyagi, Steven Li, Wenjia You, Miheer Bavare, Arda Uçar, Zhongwang Fang, Brian Jang, Bing Liu, Yunzhong He
- URL: https://arxiv.org/abs/2512.14865
- BibTeX key (if we add it): Gosai2025AudioMultiChallenge
- Tags: multi-turn, spoken-dialogue, benchmark, long-range-context, e2e-audio

## One-sentence takeaway

Audio MultiChallenge extends the text MultiChallenge framework into **audio-native, multi-turn** conversations (with a new **Voice Editing** axis) and shows current end-to-end spoken dialogue models still fail frequently on natural disfluencies, audio cues, and long-range context.

## What problem does it solve?

- Existing spoken-dialogue benchmarks often emphasize synthetic speech and/or single-turn tasks, leaving **natural multi-turn interaction** (repairs, backtracking, paralinguistic cues, long context) under-measured for end-to-end (raw-audio → response) dialogue models.
- Provides a reproducible benchmark plus rubrics to surface systematic failure modes in audio-native multi-turn settings.

## What is the core method / protocol?

- Benchmark: **Audio MultiChallenge**, building on the text-based MultiChallenge axes:
  - **Inference Memory**
  - **Instruction Retention**
  - **Self Coherence**
- Adds a new axis:
  - **Voice Editing**: robustness to **mid-utterance speech repairs and backtracking**.
- Audio-specific augmentations, e.g. **Audio-Cue** challenges for Inference Memory that require recalling **ambient sounds** and **paralinguistic signals**, not just semantics.
- Data creation: 452 conversations from 47 speakers with 1,712 instance-specific rubrics, via a hybrid **audio-native agentic + human-in-the-loop** pipeline designed to preserve natural disfluencies.

## What are the key metrics?

- Primary reported outcome: **pass rate** on instance-specific rubrics (overall and by axis).
- Analysis dimension: degradation trends with **longer audio context** (notably on Self Coherence).

## What are the main results?

- Even “frontier” models struggle: the best reported model (Gemini 3 Pro Preview (Thinking)) reaches **54.65%** pass rate.
- Failures are concentrated on the **new axes** (Voice Editing / audio-cue requirements), and **Self Coherence degrades** as audio context grows.

## How is this similar to GALILEO?

- Shares the same core concern: **multi-turn robustness** under realistic conversational dynamics.
- Reinforces the importance of measuring *trajectory failures* (e.g., coherence/retention degradation over turns) rather than single-turn accuracy.

## How is this different from GALILEO?

- Focuses on **audio-native** (E2E) spoken dialogue interaction, including paralinguistic cues and speech repairs; GALILEO (as currently framed) is primarily text/dialogue robustness under pressure and multi-turn drift/flip dynamics.
- Uses rubric pass/fail across axes rather than explicit “pressure vs evidence” decompositions (e.g., drift vs belief revision) or flip-quality metrics.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO includes explicit controlled operators (pressure vs evidence) and time-to-failure / recovery-style metrics, it can make **causal claims** about why models drift more directly than a broad rubric pass-rate benchmark.

## Where GALILEO is weaker / needs to improve

- If GALILEO aims to claim general multi-turn interaction robustness, it likely needs at least a discussion (or future work) about **audio-native** settings, where additional failure modes arise (repairs, paralinguistics, ambient cues).

## Action items for GALILEO (experiments / method / writing)

- [ ] Writing: Add a “beyond text” related-work paragraph noting that audio-native multi-turn benchmarks reveal additional failure modes (repairs/backtracking; audio cues; long-context coherence decay).
- [ ] Method idea: Consider an optional *“edit/backtrack”* operator in text conversations (user self-corrects mid-stream) as a cheap analogue to their Voice Editing axis.

## Quotes / details to potentially cite

- “We introduce Audio MultiChallenge, an open-source benchmark to evaluate E2E spoken dialogue systems under natural multi-turn interaction patterns.”
- “...we introduce a new axis Voice Editing that tests robustness to mid-utterance speech repairs and backtracking.”
- “...introducing Audio-Cue challenges... recalling ambient sounds and paralinguistic signals beyond semantic content.”
- “...even frontier models struggle... Gemini 3 Pro Preview (Thinking)... 54.65% pass rate.”
