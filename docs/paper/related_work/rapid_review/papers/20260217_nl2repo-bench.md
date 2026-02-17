# NL2Repo-Bench: Towards Long-Horizon Repository Generation Evaluation of Coding Agents

- Year: 2026
- Venue: arXiv
- Authors: Ding, Long, Pu, Zhou, Gao, Gao, He, Hou, Hu, Li, Shi, Wang, Zan, Zhang, Zhang, Chen, Cheng, Deng, Gu, Hua, Lin, Liu, Li, Pan, Peng, Qin, Shan, Tan, Xie, Wang, Yuan, Zhang, Zhao, Zhao, Zhu, Zhu, Zou, Ding, Jiao, Liu, Liu, Liu, Tao, Yang, Yang, Zhang, Chen, Huang, Zhang (et al.)
- URL: https://arxiv.org/abs/2512.12730
- BibTeX key (if we add it): nl2repo-bench-2026
- Tags: agents, coding, repository-generation, long-horizon, benchmark

## One-sentence takeaway

NL2Repo-Bench evaluates whether coding agents can build an *entire installable Python repository* from a single natural-language requirements doc, and finds today’s models still fail badly on long-horizon coherence (best <40% average test pass).

## What problem does it solve?

- Existing coding-agent benchmarks often measure short-horizon skills (single-file generation, localized edits, bugfixing) and don’t strongly test *end-to-end* repo construction where architecture, dependency management, cross-file consistency, and sustained planning matter.
- The paper targets evaluation for the “empty workspace → complete library” setting, with verification via tests.

## What is the core method / protocol?

- Task setup: agent receives a single natural-language requirements document and an empty workspace.
- Agent must autonomously:
  - design architecture / module breakdown,
  - manage dependencies,
  - implement multi-module logic,
  - produce an installable Python package.
- Scoring: primarily via test pass rates (verifiable, automated).
- Analysis: categorizes failure modes that arise over hundreds of interaction steps.

## What are the key metrics?

- Average test pass rate across tasks (and implicitly completion rate / whether a full repo is produced).
- Qualitative/diagnostic metrics via failure-mode analysis (not a single scalar, but used to explain why pass rates are low).

## What are the main results?

- Long-horizon repository generation is “largely unsolved” even for strong open- and closed-source models.
- Best reported performance is below ~40% average test pass.
- Common long-horizon failure modes include:
  - premature termination,
  - loss of global coherence,
  - fragile cross-file dependencies,
  - inadequate planning over long trajectories.

## How is this similar to GALILEO?

- Shared theme: **multi-step / long-horizon agent competence** is bottlenecked by trajectory-level failures, not just local correctness.
- The paper’s analysis of coherence loss and dependency brittleness is conceptually adjacent to “drift” / instability framing (but in software artifacts instead of beliefs/answers).

## How is this different from GALILEO?

- NL2Repo-Bench is about *software repository synthesis* (code + structure), not conversational belief/stance robustness.
- Primary outcome is **tests passing** (artifact correctness), rather than flip/consistency/drift metrics in dialogue.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO focuses on controlled pressure vs evidence conditions, it may offer cleaner causal separation of *why* a model changes behavior (pressure vs information), whereas NL2Repo-Bench is an end-to-end task where many confounders can contribute to failure.

## Where GALILEO is weaker / needs to improve

- If GALILEO aims to claim long-horizon agentic competence, it likely needs similarly *verifiable end-to-end* evaluations (tests, installability) and long-trajectory failure analysis.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider borrowing NL2Repo-Bench’s **“empty workspace → complete artifact”** framing as an auxiliary evaluation slice (even a small toy version), to strengthen any long-horizon claims.
- [ ] Add/align to their failure-mode taxonomy (premature termination, coherence loss, cross-file dependency fragility) when discussing long-horizon degradation.

## Quotes / details to potentially cite

- Problem statement (abstract): existing benchmarks “fail to rigorously evaluate the long-horizon capabilities required to build complete software systems.”
- Key result (abstract): “even the strongest agents achieve below 40% average test pass rates and rarely complete an entire repository correctly.”
- Failure modes (abstract): “premature termination, loss of global coherence, fragile cross-file dependencies, and inadequate planning over hundreds of interaction steps.”
