# Evaluating Large Language Models in Scientific Discovery

- Year: 2025
- Venue: arXiv
- Authors: Zhangde Song, Jieyu Lu, Yuanqi Du, Botao Yu, Thomas M. Pruyn, Yue Huang, et al.
- URL: https://arxiv.org/abs/2512.15567
- BibTeX key (if we add it): song2025evaluating
- Tags: scientific-discovery, evaluation, benchmark, scenario-grounded, project-level

## One-sentence takeaway

Introduces a domain-expert-defined, scenario-grounded benchmark that evaluates LLMs on both question-level and end-to-end project-level “scientific discovery” workflows, finding substantial gaps vs standard science QA benchmarks and diminishing gains from scaling/reasoning.

## What problem does it solve?

- Existing “science benchmarks” mostly test decontextualized factual/QA knowledge and miss core discovery processes (iterative reasoning, hypothesis generation, experiment/simulation design, and interpreting observations).
- As a result, they can overestimate real research usefulness and hide systematic failure modes that appear in multi-step discovery settings.

## What is the core method / protocol?

- Proposes a scenario-grounded evaluation framework across biology/chemistry/materials/physics.
- Domain experts define concrete research projects of genuine interest, then decompose each project into modular “research scenarios”; vetted questions are sampled from these scenarios.
- Two-level evaluation:
  - Question-level: accuracy on scenario-tied questions.
  - Project-level: models must (as a pipeline) propose testable hypotheses, design simulations/experiments, and interpret results.
- Runs this “two-phase scientific discovery evaluation (SDE)” on multiple SOTA LLMs and analyzes patterns.

## What are the key metrics?

- Scenario-grounded question-level accuracy.
- Project-level performance (implicitly: quality/success in hypothesis proposal, experiment/simulation design, and result interpretation; the abstract frames it as a project-level score).
- Comparative gaps vs “general science benchmarks”; analysis of scaling/reasoning returns and cross-provider weakness patterns.

## What are the main results?

- Consistent performance gap: strong performance on general science benchmarks does not translate to strong SDE performance.
- Diminishing returns from scaling model size and adding “reasoning” (as described in the abstract) in this discovery-oriented setting.
- Systematic weaknesses shared across top-tier models (across different providers).
- High variance across research scenarios: the “best” model can change depending on the project/scenario mix.
- Despite gaps, models sometimes show promise on discovery projects, suggesting guided exploration/serendipity matter.

## How is this similar to GALILEO?

- Shares the motivation that standard single-turn benchmarks can be misleading; evaluation should be closer to real workflows.
- Emphasizes scenario/protocol design and analysis of failure modes beyond aggregate accuracy.
- Highlights that model ranking can be context-dependent (GALILEO: persona/control + multi-turn; SDE: scenario/project variability).

## How is this different from GALILEO?

- Task domain: SDE is explicitly about scientific discovery projects (hypothesize/design/interpret), while GALILEO targets multi-turn susceptibility to persona pressure with ground-truth tasks.
- Ground-truth nature: GALILEO’s SSOT is “has a correct answer” and measures survival/TOF/recovery under pressure; SDE focuses on discovery workflows where evaluation can be broader than binary correctness.
- Interaction structure: GALILEO uses adversarial conversational pressure + neutral re-asking control; SDE uses scenario/project decomposition (not primarily social pressure).

## Where GALILEO is stronger / cleaner (if true)

- Clear causal contrast via Neutral Re-asking Control vs Persona pressure, enabling separation of drift vs pressure effects.
- Crisp, auditable metrics (survival/TOF/recovery) tied to ground-truth outcomes, with strong reproducibility artifacts.

## Where GALILEO is weaker / needs to improve

- Less coverage of end-to-end “discovery” workflows (hypothesis generation, experiment design, interpretation) compared to SDE-style project-level tasks.
- GALILEO currently targets a specific failure mechanism (persona pressure) rather than broader research-scaffold behaviors.

## Action items for GALILEO (experiments / method / writing)

- [ ] Related work positioning: cite SDE as evidence that decontextualized science QA benchmarks overestimate real-world research performance.
- [ ] Consider a discussion paragraph contrasting “social pressure robustness (GALILEO)” vs “project-level discovery competence (SDE)”, clarifying complementary evaluation axes.
- [ ] (Optional future) Explore adding a small “project/scenario grounding” variant of a GALILEO task (still with ground-truth where possible) to bridge to discovery-style evaluation.

## Quotes / details to potentially cite

- “prevailing science benchmarks probe decontextualized knowledge and overlook the iterative reasoning, hypothesis generation, and observation interpretation that drive scientific discovery.”
- “framework assesses models at two levels: (i) question-level accuracy on scenario-tied items and (ii) project-level performance, where models must propose testable hypotheses, design simulations or experiments, and interpret results.”
- Reports “consistent performance gap relative to general science benchmarks” and “diminishing return of scaling up model sizes and reasoning” in this setting (per abstract).
