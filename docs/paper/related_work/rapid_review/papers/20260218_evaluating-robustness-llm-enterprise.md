# Evaluating Robustness of Large Language Models in Enterprise Applications: Benchmarks for Perturbation Consistency Across Formats and Languages

- Year: 2026
- Venue: arXiv
- Authors: Tara Bogavelli; Oluwanifemi Bamgbose; Gabrielle Gauthier Melançon; Fanny Riols; Roshnee Sharma
- URL: https://arxiv.org/abs/2601.06341
- BibTeX key (if we add it): bogavelli2026enterprise-robustness-benchmark
- Tags: robustness, perturbations, multilingual, formats, enterprise, benchmark

## One-sentence takeaway

A benchmark suite for “enterprise-style” LLM tasks showing that small, non-semantic prompt perturbations (format, instruction position, multilinguality) can cause large quality drops, and robustness does not scale monotonically with model size.

## What problem does it solve?

- Enterprise deployments need consistent behavior across small variations in user input and prompt templates (typos/whitespace, different output formats like JSON/YAML, instruction re-ordering, and multilingual/cross-lingual settings).
- Prior robustness work is often narrow (few perturbation types, small academic datasets), making it less actionable for enterprise product teams.

## What is the core method / protocol?

- Construct “realistic enterprise” benchmark tasks:
  - Case summarization
  - Chat summarization
  - Q&A over provided reference docs
  - Entity slot-filling into a schema
- Create a base dataset (synthetic IT issues; then derive task inputs) and evaluate robustness under five perturbation categories:
  - General edits: whitespace, punctuation, casing, spelling, paraphrasing, tone
  - Positional: reordering sections/instructions/context within the prompt template
  - Format: change output format instructions (JSON, YAML, XML, HTML, Markdown) vs free-form
  - Multilingual: translate prompts and answer in that language
  - Cross-lingual: mix languages between context and requested output
- Define robustness as consistency between baseline output (original prompt) and perturbed prompt output:
  - Content similarity via LLM-as-a-judge (3/2/1 scale)
  - If content differs, measure delta in task-specific quality metrics (faithfulness/completeness/etc.)
- Evaluate 11 models (4B–120B+), highlighting within-family scaling vs across-family training differences.

## What are the key metrics?

- Content similarity score (LLM-as-judge; semantic consistency)
- Task quality metrics (depending on task):
  - Summarization: Faithfulness, Completeness
  - Q&A: Faithfulness, Completeness, Relevance, Citations, Conciseness
  - Slot filling: Exact-match F1, ROUGE-L (on extracted entities)
- Aggregate robustness scores reported per perturbation category (plus deltas vs baseline).

## What are the main results?

- Minor prompt perturbations can reduce performance by up to ~40 percentage points on key enterprise metrics.
- Robustness varies strongly by perturbation type:
  - Multilingual and cross-lingual perturbations are the hardest (lowest mean robustness)
  - Positional changes also cause notable degradation
  - “General” surface perturbations are comparatively more manageable
- Model size is not a reliable proxy for robustness across model families:
  - Within a family, scaling helps (example given: GPT-5 nano → full improves robustness)
  - Across families, training/data/recipe can dominate size (example: an 8B model outperforming some much larger models; another 8B being worst overall).

## How is this similar to GALILEO?

- Directly targets reliability/robustness under “benign” perturbations (format, ordering, multilingual), which is often what breaks real systems.
- Suggests evaluating not just average quality but stability under prompt/template variability—useful for any system claiming consistent behavior.

## How is this different from GALILEO?

- Focuses on benchmarking and measurement (robustness scoring suite) rather than proposing a new model/system architecture.
- Uses LLM-as-judge similarity + downstream task metrics; may not align with GALILEO’s preferred evaluation protocol (depending on how GALILEO defines consistency/grounding).

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO has a principled mechanism to enforce invariances (e.g., formatting / instruction-order invariance), that would be a clearer “solution” story than benchmark-only work.
- If GALILEO provides deterministic guarantees or stronger grounding checks, it can address failure modes beyond prompt sensitivity measurement.

## Where GALILEO is weaker / needs to improve

- If GALILEO’s evaluation currently lacks multilingual/cross-lingual and structured-format robustness, this paper highlights those as high-impact gaps.
- If GALILEO relies on a single prompt template, it may be overestimating real-world reliability.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a “prompt perturbation robustness” section to eval: whitespace/punctuation, instruction-position swaps, and output-format constraints (JSON/YAML) and report stability.
- [ ] Include multilingual + cross-lingual slices (at least 2–3 languages) as a stress test.
- [ ] In writing, explicitly argue why GALILEO should be robust to non-semantic variations; cite this as motivation and as a benchmark reference point.

## Quotes / details to potentially cite

- Definition framing: robustness as “ability to consistently generate outputs of similar quality when slight input modifications are applied.”
- Perturbation taxonomy: general / positional / format / multilingual / cross-lingual.
- Result headline: minor perturbations can cause large drops (reported up to ~40 points) and robustness is not monotonic in size across model families.
