# BaziQA-Benchmark: Evaluating Symbolic and Temporally Compositional Reasoning in Large Language Models

- Year: 2026
- Venue: arXiv
- Authors: Jiangxi Chen
- URL: https://arxiv.org/abs/2602.12889
- BibTeX key (if we add it): Chen2026BaziQA
- Tags: symbolic-reasoning, temporal, benchmark, multi-turn, structured-prompting

## One-sentence takeaway

A 200-question, professionally curated multi-turn MCQ benchmark (2021–2025 competition problems) for diagnosing LLM weaknesses in symbolic + temporally compositional reasoning, with an inference-time “Structured Reasoning Protocol” to test sensitivity to reasoning order.

## What problem does it solve?

- Existing reasoning evals mostly target standardized formal domains (math/code) or rely on anecdotal/prompt-driven “metaphysical/cultural” tests; this provides an objectively scorable, reproducible benchmark for a non-standard symbolic system with strong temporal composition demands.
- Separates (i) chart derivation / calendrical conversion from (ii) symbolic inference by providing a fixed, pre-computed chart context.

## What is the core method / protocol?

- Dataset: 200 multiple-choice questions from Global Fortune-teller Competition (Bazi) 2021–2025.
  - 8 “subjects” (fixed natal chart contexts) per year × 5 questions each → 40/year, 200 total.
  - 4-way MCQ → 25% chance baseline.
- Evaluation: multi-turn QA per subject.
  - Provide formatted chart context once, then answer 5 questions sequentially in one conversation.
- Structured Reasoning Protocol (SRP) (inference-time scaffold; no extra domain knowledge):
  1) Quantitative scan (element balance, Day Master strength, global structure)
  2) Severity grading (rank symbolic interactions under temporal context)
  3) Event mapping (map dominant signals to outcomes with precedence)

## What are the key metrics?

- Accuracy on 4-way MCQ (macro-average across years; also year-wise and domain-wise breakdowns).
- Variance / dispersion across multiple decoding runs (stability).
- Comparisons between baseline multi-turn prompting vs SRP prompting.

## What are the main results?

- Models are above chance but far from saturation (even strongest < ~50% according to paper text).
- Performance is sensitive to:
  - Temporal composition difficulty (year-wise variation; temporal localization failures)
  - Reasoning order / protocol (SRP has heterogeneous effects: helps some settings, hurts others)
- Systematic failure patterns include precise temporal localization and satisfying multiple symbolic conditions jointly.

## How is this similar to GALILEO?

- Emphasizes evaluation of compositional reasoning under a controlled protocol, and explicitly studies the effect of enforcing a reasoning *structure/order* at inference time.
- Multi-turn setting with a fixed context shared across several questions, similar to “session-level” reasoning and context accumulation analyses.

## How is this different from GALILEO?

- Domain is Bazi astrology (non-standard symbolic system) and the task is MCQ selection rather than (presumably) GALILEO’s primary task/domain.
- Focus is benchmark construction + diagnostic protocol, not proposing a new model or training method.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO operates in a more formal/transparent symbolic domain, it may offer clearer ground-truth semantics and easier-to-audit reasoning traces than culturally grounded symbolic systems.

## Where GALILEO is weaker / needs to improve

- If GALILEO’s evaluations underweight temporal composition and multi-condition symbolic judgments, this paper is a reminder that these are persistent failure modes and worth dedicated stress-tests.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add an evaluation slice explicitly targeting *temporal localization* + *multi-condition satisfaction*; report error decomposition rather than only aggregate accuracy.
- [ ] When using structured prompting/protocols, report *heterogeneous effects* (where it helps vs hurts) and analyze sensitivity to reasoning order.
- [ ] Consider multi-turn evaluation with a fixed “case context” and several related questions to measure context accumulation and consistency.

## Quotes / details to potentially cite

- “derived from 200 professionally curated, multiple-choice problems from the Global Fortune-teller Competition (2021–2025)”
- “evaluate … under a multi-turn setting” (chart context presented once; 5 questions sequentially)
- SRP steps: “Quantitative Scan … Severity Grading … Event Mapping”
- “models … outperform chance but remain far from saturation … sensitivity to temporal composition and reasoning order … failures on precise temporal localization and multi-condition symbolic judgments.”
