# SpeakerSleuth: Evaluating Large Audio-Language Models as Judges for Multi-turn Speaker Consistency

- Year: 2026
- Venue: arXiv
- Authors: Jonggeun Lee, Junseong Pyo, Gyuhyeon Seo, Yohan Jo
- URL: https://arxiv.org/abs/2601.04029
- BibTeX key (if we add it): SpeakerSleuth2026
- Tags: multi-turn, consistency, judging, audio, identity, modality-bias

## One-sentence takeaway

SpeakerSleuth shows that current large audio-language models used as *judges* are unreliable at detecting/localizing multi-turn **speaker identity drift** and often over-rely on **textual coherence** at the expense of acoustic cues.

## What problem does it solve?

- We increasingly evaluate speech/dialogue generation with “LALM-as-a-judge” setups, but it is unclear whether these models can actually judge **speaker consistency** across multi-turn conversations.
- Existing speaker-consistency checks often rely on **pairwise embedding similarity + thresholds**, which (per authors) do not provide holistic dialogue-level judgments and require brittle thresholds.

## What is the core method / protocol?

- Introduces **SpeakerSleuth**, a benchmark with **1,818 human-verified** evaluation instances from **4 datasets** (synthetic + real speech), spanning **152 speakers**, with controlled acoustic difficulty.
- Evaluates judge capability across 3 practical tasks:
  - **Detection:** decide whether a dialogue contains speaker inconsistency.
  - **Localization:** identify which specific turns are problematic.
  - **Discrimination:** choose which candidate audio best matches a target speaker among acoustic variants.
- Compares **nine** “widely-used” large audio-language models (LALMs) and embedding-based baselines.

## What are the key metrics?

- Task-specific accuracy-style measures for:
  - inconsistency **detection**
  - problematic-turn **localization**
  - candidate-audio **discrimination**
- Qualitative error patterns emphasized in the paper:
  - overly strict vs overly lenient “internal thresholds”
  - failure to localize despite detecting something is off
  - degradation when additional interlocutor turns are present (context pushes models toward text)

## What are the main results?

- LALM judges are **not reliable** for multi-turn speaker-consistency:
  - Some **overpredict inconsistency** even when the same speaker continues; others are **too lenient**.
  - Even when models detect inconsistency, they often **cannot localize** the problematic turns.
  - With more dialogue context (other interlocutors’ turns), LALMs can heavily prioritize **textual coherence**, missing even salient acoustic inconsistencies (authors mention failures like missing obvious gender switches).
- However, in **discrimination** (picking best-matching audio among variants), models perform **substantially better**, suggesting the issue is less “no acoustic capacity” and more “how the judging task prompts/conditions the model to use modalities.”
- Embedding-based approaches can be stronger for detection, but still show model-specific biases.

## How is this similar to GALILEO?

- Conceptual neighbor: evaluates **multi-turn consistency** and focuses on *judge/evaluator reliability* rather than only system performance.
- Highlights a general failure mode relevant to LLM evaluation: **the judge optimizes the wrong signal** (here: text over acoustics). This parallels concerns that LLM-based judges in dialogue settings can overweight surface coherence/agreeableness vs the targeted property (truth, consistency, calibration).

## How is this different from GALILEO?

- Different domain/modality: speaker identity consistency in **audio dialogues** rather than belief/truth/stance stability under conversational pressure.
- Does not study social pressure, persuasion, or drift-vs-revision controls; focuses on **acoustic inconsistency** and judge bias.

## Where GALILEO is stronger / cleaner (if true)

- GALILEO can likely provide clearer experimental separation between:
  - the *property being judged* (e.g., truth/stance stability) and
  - the *signals a judge might latch onto* (e.g., politeness, coherence, deference)
  via explicit controls and paired variants.

## Where GALILEO is weaker / needs to improve

- If GALILEO uses LLM-judging for multi-turn properties, this paper is a cautionary example that “judge models” can exhibit **systematic modality/feature imbalance** (not necessarily using the intended evidence stream).

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a short “**judge-signal leakage / judge bias**” paragraph in related work: SpeakerSleuth as evidence that judge models can overweight easy-to-use signals (coherence) and miss the targeted property.
- [ ] If we use judges, add a minimal audit: perturb the “easy signal” while keeping the true label fixed (analogue of adding coherent-but-wrong rationales) to test whether the judge is using the intended evidence.

## Quotes / details to potentially cite

- “We present SpeakerSleuth, a benchmark evaluating whether LALMs can reliably judge speaker consistency in multi-turn dialogues through three tasks…”
- “Models … prioritize textual coherence over acoustic cues…”
- Benchmark size: “1,818 human-verified evaluation instances … from 152 speakers.”
