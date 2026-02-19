# EconEvals: Benchmarks and Litmus Tests for Economic Decision-Making by LLM Agents

- Year: 2025
- Venue: arXiv
- Authors: Sara Fish et al.
- URL: https://arxiv.org/abs/2503.18825
- BibTeX key (if we add it): econevals2025fish
- Tags: agents, evaluation, economic-decision-making, multi-objective, robustness, self-consistency

## One-sentence takeaway

EconEvals proposes (i) interactive economic decision-making benchmarks and (ii) “litmus tests” that disentangle capability vs. preference-like tendencies (tradeoffs) and measure coherence/reliability of LLM agent choices.

## What problem does it solve?

- We lack domain-grounded, decision-theoretic evaluations for LLM *agents* that must (a) learn from environment feedback and (b) make choices with multiple conflicting objectives.
- Standard single-score benchmarks blur together: task competence, inconsistent behavior, and systematic “tendency”/tradeoff preferences (e.g., efficiency vs equality).

## What is the core method / protocol?

- **Benchmarks (interactive, learn-in-context):** derive environments from canonical economics problems:
  - Procurement
  - Scheduling
  - Pricing
  These are set up so an agent interacts with an environment, receives feedback, and must improve decisions over turns/episodes.

- **Litmus tests (stylized multi-objective tasks):** for each litmus test they define:
  - a decision task with **conflicting objectives**;
  - a **litmus score** capturing the model’s *tradeoff response*;
  - a **reliability score** capturing *coherence* of choice behavior;
  - a **competency score**: performance when the same task is converted to a *single well-specified objective* (intended to isolate “can it do the task” from “what tradeoff does it choose”).

- They additionally report validation checks via **self-consistency**, **robustness**, and **generalizability** analyses (as described in the abstract and paper outline).

## What are the key metrics?

- Benchmark score(s) for procurement/scheduling/pricing tasks (interactive environment success metrics; details live in the benchmark design sections).
- Litmus test outputs:
  - **Litmus score** (tradeoff response)
  - **Reliability score** (choice coherence)
  - **Competency score** (single-objective capability counterpart)

## What are the main results?

- Across “a broad array of frontier LLMs” (per abstract), they:
  - track capability/tendency shifts over time (model versioning);
  - extract economically interpretable behavioral patterns from choices + reasoning traces;
  - show the litmus test framework meaningfully distinguishes (i) capability from (ii) tradeoff tendencies, and can be stress-tested with self-consistency / robustness / generalization checks.

(For this rapid pass, I focused on abstract + section outline; the paper’s detailed quantitative comparisons are in the HTML/PDF.)

## How is this similar to GALILEO?

- Shares the **core GALILEO theme** of *multi-turn / interaction-grounded evaluation* rather than static single-turn QA.
- Emphasizes **robustness-style validation** (self-consistency, prompt robustness, generalizability) alongside headline scores.
- Separates *capability* from other behavioral dimensions (here: multi-objective tradeoffs + coherence), which is conceptually aligned with decomposing failure modes (e.g., inconsistency vs. sycophancy vs. drift).

## How is this different from GALILEO?

- Domain focus is **economic decision-making** (procurement/scheduling/pricing) rather than (e.g.) conversational robustness / sycophancy / social pressure.
- Litmus tests target **normative tradeoffs** (efficiency/equality, patience/impatience, collusion/competition) rather than user-alignment behaviors.
- Heavier emphasis on **agent-as-decision-maker** interacting with stylized environments, vs. dialogue-centric adversarial multi-turn settings.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO is specifically about *social* multi-turn robustness (pressure, persuasion, sycophancy), it likely has more direct constructs + datasets for those behaviors than EconEvals’ economic-objective litmus tests.
- GALILEO may offer clearer mappings from conversational prompts → failure labels (whereas multi-objective economic tasks can conflate “values” with “mistakes” without careful controls).

## Where GALILEO is weaker / needs to improve

- EconEvals’ **metric decomposition** (litmus vs reliability vs competency) is a nice pattern GALILEO could adopt more explicitly:
  - are we measuring *tendency* (e.g., agreement bias), *capability* (answering correctly), or *coherence* (stability across perturbations)?
- EconEvals appears to take **version-over-time** comparisons seriously; if GALILEO doesn’t, that’s a potential gap.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding an explicit **3-metric decomposition** for each GALILEO setting: (i) tendency score, (ii) competency/capability score, (iii) reliability/coherence score.
- [ ] Add a small “**litmus test**”-style appendix: very simple stylized tasks that isolate one tradeoff/behavior (e.g., “agreement vs accuracy” under controlled single-objective variants).
- [ ] In writing, frame GALILEO-style evaluations as measuring *behavioral response surfaces* and include **self-consistency / robustness / generalization** checks as first-class validation.

## Quotes / details to potentially cite

- Abstract framing of litmus tests and outputs:
  - “Each litmus test outputs a litmus score, which quantifies an LLM's tradeoff response, a reliability score, which measures the coherence of an LLM's choice behavior, and a competency score, which measures an LLM's capability at the same task when the conflicting objectives are replaced by a single, well-specified objective.”
- Abstract framing of benchmark domains:
  - “benchmarks derived from key problems in economics -- procurement, scheduling, and pricing -- that test an LLM's ability to learn from the environment in context.”
