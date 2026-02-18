# Revisiting the Reliability of Language Models in Instruction-Following

- Year: 2025
- Venue: arXiv (preprint)
- Authors: Jianshuo Dong; Yutong Zhang; Yan Liu; Zhenyu Zhong; Tao Wei; Chao Zhang; Han Qiu
- URL: https://arxiv.org/abs/2512.14754
- BibTeX key (if we add it): dong2025revisiting
- Tags: instruction-following, reliability, prompt-robustness, evaluation, benchmarking

## One-sentence takeaway

Even models with near-ceiling IFEval accuracy can be highly unreliable under subtle “cousin prompt” variations; the paper introduces reliable@k and IFEval++ to measure this nuance-oriented reliability and explores improvement recipes.

## What problem does it solve?

- Existing instruction-following benchmarks (e.g., IFEval) can be saturated while failing to capture a practical reliability failure mode: sensitivity to small, intent-preserving prompt changes (phrasing, context, slight constraint tweaks).
- Need a metric + benchmark that quantify *consistency across related prompts*, not just average accuracy on isolated prompts.

## What is the core method / protocol?

- Define **reliable@k**: for a group of k cousin prompts, score 1 only if the model passes *all* k prompts (a conjunctive “all-pass” reliability notion).
- Build an automated cousin-prompt generation pipeline with 3 augmentation types:
  - Rephrasing (same semantics, different wording)
  - Distractor addition (add extra compatible constraint; evaluate only original constraints)
  - Constraint/task reconfiguration (slightly change configurable constraints or reframe task while preserving requirement types)
- Add a **code-assisted validity checker** to filter flawed augmentations.
- Construct **IFEval++**: 541 base cases, each expanded to 10 prompts (1 original + 9 cousins).
- Evaluate 46 models (20 proprietary, 26 open-source) and report reliable@k across augmentation subsets.

## What are the key metrics?

- IFEval accuracy (equivalently reliable@1)
- reliable@k (notably reliable@10 on IFEval++)
- “Relative drop” from IFEval accuracy to reliable@10

## What are the main results?

- Large drops from IFEval accuracy → reliable@10 on IFEval++ (up to ~61.8% relative drop for small models; even top models drop materially).
- Nuance-oriented reliability is a **second-order property**: rankings can change compared to accuracy-only (a model can be mediocre on accuracy but comparatively strong on reliable@10).
- Reliability differs by augmentation type: rephrasing is easier; distractors and constraint/task reconfiguration degrade more.
- Explored improvement pathways:
  - Prediction-based: verbalized confidence and prompt perplexity are near-chance; probing hidden states helps but is not sufficient.
  - Training-based: fine-tuning on semantically adjacent/augmented cousin prompts can improve reliable@k; generic instruction SFT may not.
  - Test-time scaling: parallel sampling + selection (rejection sampling with a suitable checker/selector) gives the strongest gains in their study.

## How is this similar to GALILEO?

- Highlights a robustness gap between benchmark performance and real-world reliability under small specification/prompt variations.
- Provides a concrete way to operationalize “robust across paraphrases/nearby specs” as an evaluation target (reliable@k), which is broadly aligned with dependable model behavior goals.

## How is this different from GALILEO?

- Focuses narrowly on single-turn instruction-following constraints (IFEval-style verifiable constraints), rather than multi-step reasoning/agentic workflows.
- Reliability is defined as an *all-pass across k cousins* metric; depending on GALILEO’s objectives, this may be stricter than necessary (and may overweight the hardest cousin).

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO targets end-to-end task success (not just constraint satisfaction), it may better reflect user utility than constraint-only evaluation.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently reports only single-prompt accuracy (or a single canonical phrasing), it may miss the exact instability this paper quantifies.
- If GALILEO lacks systematic “cousin prompt” stress tests, it may overestimate reliability.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a “cousin prompt” evaluation slice for key tasks: paraphrase, distractor constraints, and slight parameter reconfigurations; report a reliable@k-style metric.
- [ ] When claiming reliability, include *worst-case / group-consistency* metrics (all-pass across variants), not just average accuracy.
- [ ] Consider test-time scaling baselines (e.g., sampling + verifier/selector) as a strong competitor to training changes for reliability.

## Quotes / details to potentially cite

- “Advanced LLMs have achieved near-ceiling instruction-following accuracy on benchmarks such as IFEval… [but] do not necessarily translate to reliable services in real-world use…” (Abstract)
- Introduces “nuance-oriented reliability” and the metric **reliable@k** (Abstract/Intro)
- Reports that performance “can drop by up to 61.8% with nuanced prompt modifications.” (Abstract)
