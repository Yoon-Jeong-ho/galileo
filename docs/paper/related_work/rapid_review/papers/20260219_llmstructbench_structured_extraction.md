# LLMStructBench: Benchmarking Large Language Model Structured Data Extraction

- Year: 2026
- Venue: arXiv
- Authors: Sonke Tenckhoff; Mario Koddenbrock; Erik Rodner
- URL: https://arxiv.org/abs/2602.14743
- BibTeX key (if we add it): llmstructbench2026tenckhoff
- Tags: benchmark, structured-output, json, information-extraction, robustness

## One-sentence takeaway

LLMStructBench is a 995-case JSON extraction benchmark that disentangles (a) output structural validity from (b) semantic/value correctness and shows prompting choice can dominate model size for JSON reliability.

## What problem does it solve?

- Practitioners want to use LLMs for ETL-style information extraction into a fixed JSON schema, but models often fail either by producing invalid JSON / schema-noncompliant structure or by filling the schema with wrong/missing values.
- Prior evaluations often blur these failure modes; the paper proposes a benchmark + metrics that separately capture structural validity and semantic extraction quality.

## What is the core method / protocol?

- Dataset: 995 manually verified test cases across 5 realistic "email-like" workflow scenarios (each scenario has 199 cases) with a fixed JSON schema per scenario.
  - Use-cases: Support tickets; Sick leave; Project extension; Conference registration; Loan request (includes arrays of objects).
- Generation pipeline:
  - Create fully populated ground-truth JSON objects (synthetic), then generate corresponding natural-language messages (GPT-4o used for synthesis), then manually verify and clean to ensure text contains all required values.
- Evaluation setup:
  - Model input: (i) natural-language message, (ii) an example (text, reference JSON) pair, and (iii) the JSON schema.
  - Output: a single JSON object.
  - Comparison: recursive diff vs ground truth to categorize errors.
- Prompting strategies: evaluates 5 strategies (details not fully captured in the truncated HTML fetch; paper emphasizes strategy choice).
- Model coverage: 22 open-source models + GPT-4o as a reference.

## What are the key metrics?

- Document-level validity / structural correctness (whether the output is parseable JSON and schema-compliant).
- Token/character-level and document-level measures of semantic correctness.
- Error taxonomy includes:
  - Missing Key (MK): key present in reference but absent in output.
  - Missing Value (MV): key missing or value null/absent.
  - Wrong Value (WV): value differs from reference (further subtypes discussed in the paper).

## What are the main results?

- Prompting strategy can matter more than "standard" model attributes like parameter count for achieving structurally valid JSON.
- Stronger structure-enforcing prompts can improve syntactic/structural validity (especially for smaller/less reliable models) but may increase semantic/value errors (trade-off between validity and correctness).
- Provides actionable guidance by benchmarking many open-weight models under multiple prompting styles.

## How is this similar to GALILEO?

- Shared theme: robustness/reliability of LLM outputs under constraints (schema/format adherence vs correctness), which is often the practical bottleneck in automated pipelines.
- Benchmark + metrics perspective may align with GALILEO-style evaluation framing (separating failure modes rather than using a single aggregate score).

## How is this different from GALILEO?

- Focuses specifically on JSON structured data extraction from messages (IE/ETL) and prompting strategies for schema compliance.
- Uses a synthetic-then-manually-verified dataset; may not capture distribution shift / adversarial formatting / real-world noise beyond templated variability.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO targets broader robustness/generalization, it may better cover out-of-distribution behavior beyond schema-constrained extraction.
- If GALILEO includes more realistic, naturally occurring corpora, it may better represent operational failure modes (typos, partial info, ambiguous statements, multi-intent messages).

## Where GALILEO is weaker / needs to improve

- Consider adding explicit decomposed metrics similar to LLMStructBench (validity vs semantic correctness) if GALILEO currently reports a single aggregate.
- Consider reporting prompt sensitivity (variance across prompting templates) as a first-class robustness axis.

## Action items for GALILEO (experiments / method / writing)

- [ ] In evaluation section, explicitly separate "format/schema validity" from "value correctness" and report both.
- [ ] Add an ablation that sweeps prompting strategies (e.g., single-shot schema prompt vs two-step generate-then-organize; schema-driven prompting) and quantify the trade-off between validity and semantic errors.
- [ ] Consider an error taxonomy table aligned with MK/MV/WV to make failure analysis more interpretable.

## Quotes / details to potentially cite

- Dataset scale/composition: "995 manually verified samples" across "five use cases" with "199 validated test cases" each.
- Key qualitative claim: prompting strategy choice can be more important than model size for producing structurally valid JSON, with a trade-off that stricter structure can increase semantic errors.
