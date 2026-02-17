# Stress Tests REVEAL Fragile Temporal and Visual Grounding in Video-Language Models

- Year: 2026
- Venue: arXiv
- Authors: Sethuraman T; Savya Khosla; Aditi Tiwari; Vidya Ganesh; Rakshana Jayaprakash; Aditya Jain; Vignesh Srinivasakumar; Onkar Kishor Susladkar; Srinidhi Sunkara; Aditya Shanmugham; Rakesh Vaideeswaran; Abbaas Alif Mohamed Nishar; Simon Jenni; Derek Hoiem
- URL: https://arxiv.org/abs/2602.11244
- BibTeX key (if we add it): reveal2026stress
- Tags: robustness, diagnostic-benchmark, video-language-models, temporal-grounding, motion, sycophancy

## One-sentence takeaway

REVEAL is a controlled stress-test suite showing that many video-language models over-rely on language priors and user assertions—failing basic temporal direction, camera motion, occlusion integration, and even *video sycophancy* checks.

## What problem does it solve?

- Standard VidLM benchmarks can be “solved” via non-visual shortcuts (language priors, dataset biases), so high accuracy may not reflect true visual/temporal grounding.
- There is a lack of **diagnostic**, controlled tests that isolate *why* a VidLM fails (visual evidence use vs temporal reasoning vs motion understanding), not just whether it fails.

## What is the core method / protocol?

- Introduces **REVEAL**, a diagnostic benchmark organized as **five controlled stress tests** grouped into three failure modes:
  - **Insufficient use of visual evidence**
    - *Language-only shortcuts*: evaluate whether models answer correctly even when video is shuffled or even blank (black frames), indicating reliance on text priors.
    - *Video sycophancy*: present assertive user claims that are **plausible but visually false**, then measure agreement in a T/F setup.
  - **Weak temporal understanding**
    - *Temporal expectation bias*: reverse videos, remove segments, or inject temporal/spatial anomalies and check whether models describe what’s actually shown vs what’s “expected”.
    - *Spatiotemporal occlusion*: mask complementary regions across successive frames so full-scene recovery requires cross-frame integration.
  - **Limited motion awareness**
    - *Camera motion sensitivity*: synthetic camera-motion videos (pan/tilt/zoom) and driving lane-change style tasks.
- Emphasis is on **controlled transformations** of existing datasets + a pipeline to auto-generate diagnostic examples.

## What are the key metrics?

- Task accuracy / agreement rates per stress test (multi-dimensional “capability profile” rather than a single score).
- For video sycophancy specifically: fraction of times the model agrees with a **false** user claim (sycophancy rate).

## What are the main results?

- Across leading open/closed VidLMs, REVEAL reports consistent brittleness:
  - models describe reversed scenes as forward (temporal priors dominate),
  - answer questions while neglecting video content (language-only shortcuts),
  - agree with false claims (video sycophancy),
  - struggle with basic camera motion,
  - and fail to integrate temporally distributed evidence under simple masking.
- Humans reportedly succeed easily on these same controlled tests.

## How is this similar to GALILEO?

- Same core thesis: **single headline accuracy can hide fragility**; diagnostic evaluations should isolate *failure modes*.
- The *sycophancy* stress test parallels GALILEO-style “pressure vs evidence” tension: models may comply with an assertive user claim even when the underlying evidence contradicts it.
- Multi-dimensional reporting (separating distinct weaknesses) matches GALILEO’s need to disentangle different mechanisms (e.g., drift vs correction).

## How is this different from GALILEO?

- Domain: multimodal **video-language** grounding vs GALILEO’s focus on text-centric multi-turn robustness / belief stability.
- Interaction structure: REVEAL is primarily controlled *single-instance* diagnostics (though about temporal sequences), not long-horizon multi-turn dialogue trajectories with recovery dynamics.
- Failure event: mostly “wrong answer / wrong grounding” rather than “turn-of-failure under social pressure” with survival-style analysis.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO uses explicit *multi-turn* protocols with **time-to-failure / recovery** metrics, it can better characterize longitudinal dynamics than REVEAL’s per-test snapshots.
- GALILEO’s pressure operators and controls can more directly separate *evidence-driven updating* from *pressure-driven drift* in text settings.

## Where GALILEO is weaker / needs to improve

- If we discuss sycophancy/robustness broadly, multimodal evidence like REVEAL is a reminder that “over-trusting priors / agreeing with users” is cross-modal; we may need a short positioning note clarifying scope (text agents) while acknowledging multimodal parallels.

## Action items for GALILEO (experiments / method / writing)

- [ ] Related work: cite REVEAL as evidence that **sycophancy + shortcut reliance** also appears in VidLMs, supporting the claim that “apparent competence” can be an illusion under biased prompts.
- [ ] Writing: add a sentence drawing an analogy between REVEAL’s “language-only shortcut” and text-only multi-turn settings where models may follow conversational priors instead of evidence.

## Quotes / details to potentially cite

- Abstract: introduces “REVEAL, a diagnostic benchmark … five controlled stress tests; assessing temporal expectation bias, reliance on language-only shortcuts, video sycophancy, camera motion sensitivity, and robustness to spatiotemporal occlusion.”
