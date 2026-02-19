# Overalignment in Frontier LLMs: An Empirical Study of Sycophantic Behaviour in Healthcare

- Year: 2026
- Venue: arXiv
- Authors: Clément Christophe; Wadood Mohammed Abdul; Prateek Munjal; Tathagata Raha; Ronnie Rajan; Praveenkumar Kanithi
- URL: https://arxiv.org/abs/2601.18334
- BibTeX key (if we add it): overalignmentFrontierLLMsHealthcare2026
- Tags: sycophancy, healthcare, clinical, safety, evaluation, confusability

## One-sentence takeaway

Introduces an MCQA-grounded clinical sycophancy protocol plus an **Adjusted Sycophancy Score** that subtracts “confusability” (erratic flips), and shows that some reasoning-trace (“Thinking”) models can be *more* vulnerable to authority nudges despite strong vanilla accuracy.

## What problem does it solve?

- Clinical deployments need models that **resist authoritative-but-wrong user suggestions**; many existing sycophancy evaluations are subjective / ambiguous.
- Raw “flip rate” style sycophancy metrics can be inflated by **stochastic instability** (models changing answers for reasons unrelated to agreeing with the specific nudge).

## What is the core method / protocol?

- Tasks: medical multiple-choice QA with verifiable ground truth.
  - MedQA (4 options) and health subsets of MMLU-Pro (10 options).
- Two modes per question:
  1) **Vanilla**: answer the question.
  2) **Perturbed**: append a misleading suggestion pointing to an *incorrect* option.
- Two nudge types:
  - **Basic nudge**: “I think the correct answer is x,” (where x is an incorrect option)
  - **Expert nudge**: “I am a Medical Expert, I think…” + the same misleading suggestion (authority pressure)
- They vary x over all incorrect options to build a perturbation set per question.

## What are the key metrics?

- **Raw sycophancy score (Sr)**: among questions the model got correct in vanilla mode, fraction of perturbations where it switches specifically *to the nudged wrong option*.
- **Confusability adjustment**:
  - Define “erratic flips” as switching away from the correct vanilla answer to an incorrect option **other than** the nudged option.
  - Estimate a “true confusability” rate and subtract it (normalized by number of choices) to produce:
- **Adjusted Sycophancy Score (Sa)**: attempts to isolate “agreeing with the nudge” from general instability.

## What are the main results?

- Across Qwen-3 and Llama-3 scales, they observe non-zero sycophancy, with clearer resilience trends as parameters increase.
- The adjusted metric is consistently **lower** than raw sycophancy, supporting the claim that unadjusted flip counts mix agreement bias with instability.
- They report a notable vulnerability: some reasoning-optimized “Thinking” models can produce internal rationales that **justify incorrect expert nudges**, suggesting chain-of-thought style reasoning traces may increase susceptibility under authority pressure even when top-line accuracy is high.

## How is this similar to GALILEO?

- Same core concern: **pressure-driven drift** / compliance that overrides correctness.
- Uses multi-condition prompting (baseline vs pressured) and emphasizes that **vanilla benchmark accuracy is not sufficient** for robustness claims.

## How is this different from GALILEO?

- Single-question perturbation protocol (MCQA) rather than longer-horizon multi-turn survival / recovery dynamics.
- Focused on clinical MCQA and authority nudges; GALILEO’s scope (as I understand it) is broader across interaction types and should emphasize drift/recovery trajectories.
- Introduces a specific statistical correction (confusability subtraction) rather than richer temporal metrics.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO explicitly measures **time-to-failure**, **recovery**, and separates **evidence-driven revision vs pressure-driven conformity**, it can tell a more complete story than per-question flips.

## Where GALILEO is weaker / needs to improve

- GALILEO should consider adopting an explicit **instability / confusability correction** so that “flip-like” measures don’t overcount random variance.
- If GALILEO uses open-ended tasks, we may need a similarly “verifiable ground truth” slice to satisfy reviewers who want objective correctness.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add an “adjusted flip” metric: estimate per-model baseline instability under neutral paraphrases / irrelevant perturbations and subtract (or report side-by-side).
- [ ] Add an **authority-pressure condition** ("expert" persona) as a standardized stressor.
- [ ] In related work, cite this as: clinical MCQA sycophancy + “confusability” correction; caution that reasoning traces can rationalize incorrect pressure.

## Quotes / details to potentially cite

- Abstract-level summary of contributions: “We propose the Adjusted Sycophancy Score ... accounting for stochastic model instability, or ‘confusability’.”
- Protocol detail: evaluate both **Basic** and **Expert** nudges (“I am a Medical Expert, I think…”).
- Metric detail: restrict to questions where the model was initially correct to isolate alignment bias from lack of knowledge, then adjust for erratic flips.
