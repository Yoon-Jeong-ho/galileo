# CheckEval: Robust Evaluation Framework using Large Language Model via Checklist

- Year: 2024
- Venue: CHI 2024 Workshop on Human-centered Evaluation and Auditing of Language Models
- Authors: (not captured in arXiv HTML excerpt)
- URL: https://arxiv.org/html/2403.18771v1
- BibTeX key (if we add it): CheckEval2024
- Tags: evaluation, llm-judge, checklists, robustness, interpretability

## One-sentence takeaway

CheckEval replaces fuzzy Likert-style LLM judging with a decomposed, boolean checklist (Yes/No questions) whose positive-rate is aggregated into a score, improving interpretability and inter-annotator consistency.

## What problem does it solve?

- LLM-as-judge evaluations can be ambiguous and prompt-sensitive; Likert scales make it hard to interpret score differences and to reproduce consistent judgments (for both humans and LLMs).
- Need an evaluation protocol that is (i) more interpretable, (ii) more robust/consistent across evaluators, and (iii) customizable to task-specific aspects.

## What is the core method / protocol?

- Three-stage framework:
  1) **Aspect selection (human-driven):** choose evaluation aspects (e.g., fluency/consistency) and define **key components** for each aspect.
  2) **Checklist generation:**
     - (a) write one **key boolean question** per key component,
     - (b) **augment** each key question into more specific boolean questions using an LLM (they mention GPT-4),
     - (c) **filter** questions for clarity, redundancy, and alignment (authors curate; typical retention yields ~3–5 questions per key component).
  3) **Checklist-based evaluation:** have an LLM answer each checklist question Yes/No for a generated text; **aggregate score = proportion of “Yes”** answers.

## What are the key metrics?

- Correlation with human judgments on SummEval (they mention Spearman’s ρ and Kendall’s τ in the case study section / table header).
- Inter-Annotator Agreement (IAA) consistency across evaluator models (described qualitatively in the introduction/abstract; specific coefficient not captured in excerpt).

## What are the main results?

- Case study on SummEval (summarization):
  - They sample **10%** of SummEval (to match human-annotation distribution across aspects) and evaluate across the four SummEval aspects.
  - Report **strong correlation** with human judgments and **highly consistent IAA** among LLM evaluators when using the checklist protocol (exact numbers not captured in the fetched excerpt).

## How is this similar to GALILEO?

- Shares the general motivation of making evaluation **more reliable and interpretable**, by structuring what is otherwise a subjective judgment.
- Decomposition into sub-aspects resembles “break the objective into atomic checks” thinking (similar spirit to atomic-fact / fine-grained evaluation lines of work).

## How is this different from GALILEO?

- CheckEval is primarily a **protocol for LLM-based evaluation** (checklist + boolean QA + aggregation), not a new model or training method.
- The checklist construction includes **human aspect/key-component definition** and **manual filtering**, which may not scale or may introduce author bias.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO targets end-to-end evaluation with fewer manual steps, it may avoid the manual “question filtering” bottleneck and reduce potential evaluator-design bias.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently uses single-score LLM judging (Likert / direct scalar), this paper suggests a concrete path to improve **robustness and explainability** via decomposed boolean checks.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a checklist-based evaluation variant: decompose each aspect into key components + boolean questions; compare stability vs direct scalar judging.
- [ ] Report not just correlation with humans but also **IAA across evaluator prompts/models** (before/after checklist).
- [ ] Try alternative aggregation beyond mean-Yes (e.g., weighted components, minimum-per-component) and see if it improves sensitivity.

## Quotes / details to potentially cite

- “CheckEval decomposes evaluation criteria into more detailed sub-aspects and develops a checklist for each dimension… breaks down the evaluation into discrete, Boolean questions… prompts LLMs to respond to the checklist… aggregates these responses to compute a final score.” (paraphrased from Intro / Design)
- Checklist generation pipeline: key questions → LLM augmentation (GPT-4 mentioned) → manual filtering; typically retain “3–5 questions per key component” and “average of 4 questions per component.”
- Score aggregation: “proportion of positive responses (‘Yes’ answers) to the total number of questions used as the final score.”
