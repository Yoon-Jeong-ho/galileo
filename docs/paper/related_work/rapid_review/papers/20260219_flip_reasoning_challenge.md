# FLIP Reasoning Challenge

- Year: 2025
- Venue: arXiv
- Authors: Andreas Plesner; Turlan Kuzhagaliyev; Roger Wattenhofer
- URL: https://arxiv.org/html/2504.12256
- BibTeX key (if we add it): plesner2025flip
- Tags: multimodal reasoning; visual storytelling; benchmark/dataset; human-verification

## One-sentence takeaway

FLIP is a multimodal benchmark of 4-image “choose the coherent story ordering” challenges (from Idena human-verification) that remains difficult for SOTA VLMs/LLMs, exposing a gap to human sequential commonsense reasoning.

## What problem does it solve?

- Provides a reasoning-focused alternative to “perception-only” CAPTCHAs/benchmarks, targeting sequential/commonsense visual reasoning rather than object recognition.
- Offers a dataset with strong human agreement (ground truth via blockchain consensus voting) and a simple evaluation protocol.

## What is the core method / protocol?

- Task: each example contains 4 images shown in two different orderings (“left” vs “right” stack); predict which ordering forms a coherent story.
- Data source: FLIP challenges from the Idena blockchain; uses aggregated human votes + consensus scoring; filters out “no consensus.”
- Evaluation: accuracy of selecting correct stack; they report results for many models and also ensembles.
- Notable modeling finding: generating captions for images and then reasoning over the text can outperform direct VLM reasoning on raw images.

## What are the key metrics?

- Accuracy on the binary choice (left vs right coherent ordering), reported in zero-shot settings (and with variants such as caption-assisted pipelines).

## What are the main results?

- Human accuracy: 95.3% (84,600 participant answers).
- Best single-model (reported):
  - Open-source max accuracy: 75.5% (zero-shot).
  - Closed-source max accuracy: 77.9% (zero-shot).
- Captioning + reasoning can improve performance vs raw-image prompting for some systems (example reported: Gemini 1.5 Pro improves from 69.6% to 75.2% when using caption text).
- Ensembling 15 models increases accuracy to 85.2% (still below human).

## How is this similar to GALILEO?

- Relevant as related work if GALILEO targets (or claims) robust reasoning evaluation: FLIP is a concrete benchmark emphasizing sequential coherence and commonsense in a multimodal setting.
- Illustrates a recurring theme: converting perception into structured/textual intermediates (captions) can improve downstream reasoning—useful precedent for “representation then reason” pipelines.

## How is this different from GALILEO?

- FLIP is explicitly a human-verification-derived dataset (blockchain tasks) with a very specific interaction format (two permutations of 4 images).
- Focuses on accuracy on a fixed benchmark rather than building a general method; improvements are mostly via prompting/captioning/ensembles rather than new training objectives.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO includes principled task generation and controlled difficulty calibration, it may offer clearer diagnostics than the fixed 4-image ordering format.
- If GALILEO supports broader compositionality and richer supervision than binary choice, it can avoid ceiling effects and better attribute failure modes.

## Where GALILEO is weaker / needs to improve

- If GALILEO lacks a “human-grounded” consensus signal (like Idena voting) or a large-scale naturally occurring distribution, it may be more vulnerable to synthetic artifacts or limited diversity.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider citing FLIP as a multimodal sequential-reasoning benchmark that remains challenging even for strong proprietary models.
- [ ] If GALILEO uses intermediate representations, reference FLIP’s caption-then-reason finding as motivation.
- [ ] Add a brief comparison paragraph: why GALILEO’s tasks/format provide complementary stress-tests vs 4-image story ordering.

## Quotes / details to potentially cite

- “FLIP challenges present users with two orderings of 4 images, requiring them to identify the logically coherent one.”
- “Even the best open-sourced and closed-sourced models achieve maximum accuracies of 75.5% and 77.9%, respectively, in zero-shot settings, compared to human performance of 95.3%.”
- “Combining the predictions from 15 models in an ensemble increases the accuracy to 85.2%.”
