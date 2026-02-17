# Evaluating the Sensitivity of LLMs to Prior Context

- Year: 2025
- Venue: arXiv
- Authors: Robert Hankache; Kingsley Nketia Acheampong; Liang Song; Marek Brynda; Raad Khraishi; Greig A. Cowan
- URL: https://arxiv.org/abs/2506.00069
- BibTeX key (if we add it): khraishi2025priorcontext (suggested)
- Tags: multi-turn, context-sensitivity, robustness, evaluation, placement-effects, GPQA

## One-sentence takeaway

Even strong frontier LLMs show large, sometimes non-monotonic accuracy drops on a fixed multiple-choice QA task when preceded by long and/or domain-mismatched prior context, and prompt placement (where the task description appears) can mitigate the drop substantially.

## What problem does it solve?

- Standard single-turn QA benchmarks (e.g., GPQA/MMLU-style) do not capture *contextual sensitivity* in realistic multi-turn deployments.
- We need controlled ways to quantify how *prior turns* (length, content type, domain coherence) affect downstream task accuracy.

## What is the core method / protocol?

- Construct new benchmarks derived from GPQA Diamond (Biology/Physics/Chemistry multiple-choice questions) where the *target query* is held fixed but is preceded by an *optional prior context* intended to simulate conversation history.
- Systematically vary:
  - volume/length of prior context
  - nature of prior context (e.g., domain-consistent vs domain-shift / less helpful context)
  - placement of the task description within the long context (prompt-position effects)
- Evaluate multiple closed models (GPT/Claude/Gemini families) using accuracy on the MCQ as the primary outcome.

## What are the key metrics?

- Multiple-choice accuracy on GPQA-derived questions.
- Relative degradation vs a no-prior-context baseline (reported as percentage-point or percent drops).
- Sensitivity/mitigation effect sizes from prompt placement changes.

## What are the main results?

- Prior context can induce dramatic performance degradation on MCQ QA, reported up to ~73% drop for some models.
- Even highly capable models (e.g., GPT-4o) show sizable drops (reported up to ~32% decrease in accuracy).
- Scaling is not reliably monotonic: larger models are not always predictably more robust than smaller ones under prior-context perturbations.
- Prompt engineering via *strategic placement* of the task description can mitigate context-induced drops; they report improvements up to ~3.5× accuracy in some settings.

## How is this similar to GALILEO?

- Same broad objective: stress-test model reliability under *multi-turn / long-context* conditions rather than single-turn snapshots.
- Highlights that *history / prior turns* are a first-class axis of robustness (not just question difficulty).
- Provides a clean example of an evaluation knob we can cite: systematic manipulation of prior context length/type and measurement of degradation.

## How is this different from GALILEO?

- Focuses on *MCQ accuracy* on GPQA-style STEM questions; GALILEO is centered on dialogue robustness failure dynamics (and, in our framing, phenomena like drift/instability under interaction pressures).
- Perturbations are primarily *context volume/domain/placement*, not explicitly adversarial social pressure, persuasion, or user steering.
- Does not appear to emphasize time-to-event / survival-style metrics; it is more “before/after context” accuracy comparisons.

## Where GALILEO is stronger / cleaner (if true)

- If we explicitly model failure over turns (e.g., turn-of-failure / survival curves), we provide a richer temporal characterization than aggregate accuracy drops.
- If we include control conditions separating evidence-driven updates from spurious drift, we can make stronger causal claims about what drives degradation.

## Where GALILEO is weaker / needs to improve

- We should more explicitly report *prompt-position / placement* sensitivity as a confound and as a mitigation lever (this paper suggests it can be large).
- We may need a clearer “context type” taxonomy (domain-consistent vs domain-shift vs noisy/irrelevant) and show robustness across these.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add an ablation where the task instruction/format spec is placed at different positions (start vs middle vs end) in long contexts, and quantify robustness impact.
- [ ] Add a “domain-shift prior turns” condition as a non-adversarial stressor to disentangle general context overload from adversarial drift.
- [ ] In related work, cite this as evidence that single-turn QA is not a proxy for long-horizon reliability; motivate multi-turn evaluation.

## Quotes / details to potentially cite

- They report that performance on multiple-choice questions “can degrade dramatically in multi-turn interactions,” with drops “as large as 73% for certain models,” and that prompt placement can mitigate drops, improving accuracy “by as much as a factor of 3.5.”
