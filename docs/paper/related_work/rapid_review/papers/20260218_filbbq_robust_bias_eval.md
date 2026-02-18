# Robust Bias Evaluation with FilBBQ: A Filipino Bias Benchmark for Question-Answering Language Models

- Year: 2026
- Venue: LREC 2026 (accepted; arXiv preprint)
- Authors: Lance Calvin L. Gamboa; Yue Feng; Mark Lee
- URL: https://arxiv.org/abs/2602.14466
- BibTeX key (if we add it): Gamboa2026FilBBQ
- Tags: bias, fairness, robustness, evaluation, multilingual, Filipino, QA

## One-sentence takeaway

FilBBQ extends the BBQ bias benchmark to Filipino with >10k prompts and proposes a more reliable evaluation protocol by averaging bias scores across multiple random seeds to reduce instability.

## What problem does it solve?

- Lack of BBQ-style bias evaluation resources for low(er)-resource languages / non-Western sociocultural contexts (here: Filipino / Philippine context).
- Prior BBQ evaluations are potentially unreliable because they typically compute bias metrics from a single generation per prompt (ignoring response instability).

## What is the core method / protocol?

- Dataset/benchmark construction (4 phases):
  - Template categorization
  - Culturally aware translation
  - New template construction (Philippine-context stereotypes)
  - Prompt generation
- FilBBQ contains >10,000 prompts focused on sexist and homophobic biases relevant to the Philippines.
- Robust evaluation protocol for generative QA bias:
  - Run each prompt multiple times with different random seeds
  - Compute bias scores per seed/run
  - Average scores across seeds to reduce variance from stochastic decoding / instability

## What are the key metrics?

- BBQ-style bias scores aggregated over prompts (paper emphasizes variance across seeds and reporting averaged scores).
- Stability/variability analysis across seeds (qualitative + quantified via differing bias scores across runs).

## What are the main results?

- Bias scores vary meaningfully across different seeds for the same model/prompt set, confirming instability concerns.
- Evaluated Filipino-trained models exhibit sexist and homophobic biases in categories including:
  - emotion
  - domesticity
  - stereotyped queer interests
  - polygamy

## How is this similar to GALILEO?

- Shared theme: rigorous evaluation methodology where naïve single-run metrics can be misleading.
- Emphasizes robustness/variance considerations (repeat runs, aggregate) rather than trusting one-off outputs.

## How is this different from GALILEO?

- Focuses on social bias benchmarking for generative QA in a specific language/cultural context (Filipino/Philippines), not GALILEO’s core target.
- Primary contribution is benchmark + evaluation protocol, not a new model/system.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO already uses repeated trials / confidence intervals / robustness sweeps, it may generalize this paper’s “multi-seed averaging” principle across more dimensions than just seed.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently reports single-run results for any stochastic component, this paper is an additional citation/motivation to:
  - repeat across seeds
  - report mean + dispersion (std/CI)
  - avoid over-claiming from one run

## Action items for GALILEO (experiments / method / writing)

- [ ] Add/strengthen a “robustness to randomness” subsection: run key evaluations across multiple seeds and report mean±std (or CI).
- [ ] If relevant, cite FilBBQ as evidence that bias/behavioral metrics can be seed-sensitive and should be averaged.

## Quotes / details to potentially cite

- FilBBQ is built via a “four-phase development process consisting of template categorization, culturally aware translation, new template construction, and prompt generation.”
- The benchmark includes “more than 10,000 prompts” targeting “sexist and homophobic prejudices relevant to the Philippine context.”
- Robust protocol: “obtaining prompt responses across multiple seeds and averaging the bias scores calculated from these distinctly seeded runs.”
