# Continuous Benchmark Generation for Evaluating Enterprise-scale LLM Agents

- Year: 2025
- Venue: arXiv
- Authors: Divyanshu Saxena, Rishikesh Maurya, Xiaoxuan Ou, Gagan Somashekar, Shachee Mishra Gupta, Arun Iyer, Yu Kang, Chetan Bansal, Aditya Akella, Saravan Rajmohan
- URL: https://arxiv.org/abs/2511.10049
- BibTeX key (if we add it): saxena2025continuous
- Tags: agents, evaluation, benchmarks, enterprise, benchmark-generation

## One-sentence takeaway

Instead of static benchmarks, the paper proposes a **continuous benchmark generation pipeline** for enterprise agents, grounded in (i) semi-structured developer knowledge-base docs and (ii) “gold” reference commits, to keep evaluation aligned with evolving requirements.

## What problem does it solve?

- Enterprise agents face **non-stationary requirements** (platform/policy/dependency changes) and **heterogeneous artifacts** (code, configs, manifests, docs), making static benchmarks brittle and costly to maintain.
- Ground truth is often sparse/expensive: “correctness” may require reconstructing intended outcomes across many changing components.

## What is the core method / protocol?

- Proposes a pipeline with two key ingredients:
  - **Developer-authored semi-structured KB documents** describing task intent, context, and expected outcomes (one per subtask), which can be edited as requirements evolve.
  - **Reference implementations** derived from a small number of real changes (e.g., commits/PRs) linked to KB docs to produce benchmark ground truths.
- Instantiated on a case study of **service migration** between infrastructure platforms (the paper notes this involved large-scale heterogeneous changes such as build scripts, dependencies, and deployment files).

## What are the key metrics?

- The excerpted sections emphasize *process + maintainability* rather than a specific fixed metric suite.
- Implied evaluation outputs:
  - Fine-grained, per-subtask metrics (since KBs are per-subtask / per-artifact class)
  - Longitudinal tracking as requirements/agents change

## What are the main results?

- Conceptual contribution + case-study motivation: shows why **static benchmarks** are insufficient for enterprise migration-like settings and argues their pipeline yields a **maintainable evaluation framework** enabling rapid feedback and targeted improvements.
- (Rapid read note: the fetched portion does not include detailed quantitative results/ablations; revisit full paper if we decide to cite.)

## How is this similar to GALILEO?

- Shared emphasis on evaluation under **distribution shift / evolving conditions** (agent changes + environment changes).
- Similar “evaluation should reflect process/trajectory” motivation: enterprise tasks require more than final-answer accuracy.

## How is this different from GALILEO?

- This paper is about **how to keep benchmarks updated** (benchmark generation/maintenance), not about a specific **interaction protocol** (e.g., pressure vs evidence, drift vs revision) or novel metrics for multi-turn behavior.
- Oriented to software/ops enterprise workflows (migration) rather than GALILEO’s target phenomena.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO has a fixed, well-specified experimental protocol + metrics, it can provide more **scientifically controlled** comparisons than an evolving benchmark pipeline.

## Where GALILEO is weaker / needs to improve

- If GALILEO claims robustness in “real-world” settings, we may need a clearer story for **benchmark maintenance** as tasks change (this paper’s central point).

## Action items for GALILEO (experiments / method / writing)

- [ ] Related-work paragraph: cite as motivation that **static benchmarks break** in enterprise/non-stationary settings; position GALILEO’s evaluation protocol as complementary (orthogonal axis: *what to measure* vs *how to keep tasks current*).
- [ ] Consider adding a “benchmark evolution” note: if our tasks are curated, describe how we’ll version/update them (to pre-empt reviewer concerns).

## Quotes / details to potentially cite

- Abstract-level motivation: enterprise-scale agents have continuously evolving services/requirements and sparse ground truth; they propose evolving benchmarks via KB docs + gold commits.
- Intro: static benchmarks are rigid and hard to maintain under changing requirements; evaluation must be ongoing as both agents and platforms evolve.
