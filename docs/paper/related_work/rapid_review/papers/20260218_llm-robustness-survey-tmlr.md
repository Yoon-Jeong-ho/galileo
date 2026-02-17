# Robustness in Large Language Models: A Survey of Mitigation Strategies and Evaluation Metrics

- Year: 2025
- Venue: TMLR (accepted; arXiv preprint)
- Authors: Pankaj Kumar; Subhankar Mishra
- URL: https://arxiv.org/abs/2505.18658
- BibTeX key (if we add it): kumar2025robustnessllm_survey
- Tags: robustness, survey, evaluation, mitigation

## One-sentence takeaway

A broad survey that organizes LLM robustness into definitions/dimensions, sources of brittleness, mitigation strategies, and the benchmark+metric landscape, highlighting that accuracy-only evaluation misses real-world reliability.

## What problem does it solve?

- Provides a single taxonomy/overview for “robustness of LLMs”, including:
  - what robustness means for LLM behavior (vs clean benchmark accuracy)
  - where non-robustness comes from (data/training/architecture/inference/adversary)
  - how robustness is mitigated (pre-/in-/intra-processing, test-time adaptation)
  - how it is evaluated (benchmarks, emerging metrics, gaps)

## What is the core method / protocol?

- Survey paper (not a new algorithm).
- Organizes prior work into sections roughly:
  - robustness definitions + key dimensions
  - sources of non-robustness (data-, training-, architectural-, inference-related)
  - mitigation strategies:
    - pre-processing (augmentation, filtering)
    - in-processing (adversarial training, regularization, alignment)
    - intra-processing (robust prompting, decoding modifications, weight redistribution)
    - inference-time adaptation / test-time compute
  - evaluation landscape: benchmarks/metrics and remaining gaps

## What are the key metrics?

- Discusses “beyond-accuracy” robustness evaluation in general terms (survey).
- Emphasizes that robustness should measure stability under perturbations, distribution shift, noisy/unstructured inputs, and adversarial inputs.

## What are the main results?

- Not an empirical contribution; main “result” is a structured synthesis.
- Clear framing that high benchmark accuracy is insufficient for reliability; robustness evaluation needs explicit perturbation/shift/adversarial testing and better metrics.

## How is this similar to GALILEO?

- Overlaps with GALILEO’s motivation: evaluation should capture reliability/stability under perturbations and realistic failure modes, not just aggregate accuracy.
- Useful as a citation hub for related-work framing (definitions, categories of brittleness, families of mitigation).

## How is this different from GALILEO?

- Survey-level: does not propose a concrete experimental protocol tailored to multi-turn “drift vs evidence-based revision”, nor a specific metric suite.
- Robustness scope is broad (includes adversarial attacks, training/inference mitigations); GALILEO appears more focused on behavioral stability/revision under conversational pressure.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO provides a crisp, operational robustness metric/protocol (e.g., time-to-failure / survival-style metrics; return-to-truth after misinformation), it can be positioned as a concrete, testable protocol compared to survey taxonomies.

## Where GALILEO is weaker / needs to improve

- Related-work breadth: GALILEO should ensure its introduction situates itself relative to the larger robustness taxonomy (data/training/decoding/prompting mitigations) even if it focuses on evaluation.

## Action items for GALILEO (experiments / method / writing)

- [ ] Cite this survey in the intro/background when motivating robustness as “beyond accuracy” + enumerating robustness dimensions.
- [ ] Use its taxonomy as a checklist to state what GALILEO covers (evaluation/protocol) vs explicitly does not cover (e.g., adversarial training).

## Quotes / details to potentially cite

- Abstract framing (paraphrase): robustness requires consistent performance across diverse inputs; failures have real-world implications; and evaluation must go beyond standard accuracy.
