# TheAgentCompany: Benchmarking LLM Agents on Consequential Real World Tasks

- Year: 2024
- Venue: arXiv (preprint)
- Authors: Frank F. Xu, Yufan Song, Boxuan Li, Yuxuan Tang, Kritanjali Jain, Mengxue Bao, Zora Z. Wang, Xuhui Zhou, Zhitong Guo, Murong Cao, Mingyang Yang, Hao Yang Lu, Amaad Martin, Zhe Su, Leander Melroy Maben, Raj Mehta, Wayne Chi, Lawrence Jang, Yiqing Xie, Shuyan Zhou, Graham Neubig
- URL: https://arxiv.org/abs/2412.14161
- BibTeX key (if we add it): TheAgentCompany-Xu-2024
- Tags: agents, benchmark, workplace, tool-use, long-horizon, evaluation, reproducibility

## One-sentence takeaway

A self-hostable “software company” benchmark (175 tasks) that evaluates end-to-end LLM agents across web/coding/terminal + coworker communication, reporting both full completion and checkpoint-based partial credit (best baseline ~30% full completion).

## What problem does it solve?

- Existing agent benchmarks either (a) are not strongly tied to professional work, (b) cover narrow task types (e.g., only code patches), or (c) lack reproducible, resettable execution environments.
- The paper targets measuring real-world “digital worker” capability: browsing internal tools, writing/running code, and coordinating via workplace communication, including long-horizon tasks.

## What is the core method / protocol?

- Construct a simulated startup environment (“TheAgentCompany”) with:
  - A local sandboxed workspace (Docker) providing typical developer tools (browser + terminal + editor).
  - A self-hosted intranet composed of open-source enterprise tools (e.g., GitLab-like repo/wiki, drive/docs, issue tracker, chat).
  - Simulated coworkers: LLM-driven colleague agents that can be messaged to obtain missing info needed for tasks.
- Task design:
  - Each task includes an intent description, a set of checkpoints (intermediate milestones with point values), and programmatic evaluators that inspect environment state and/or trajectories.
  - Scoring includes both binary “task complete” and a partial-credit metric based on achieved checkpoints.
- Baseline evaluation harness:
  - Run multiple LMs through a standardized agent framework capable of browsing + coding actions (they report using OpenHands as the harness).

## What are the key metrics?

- Full task completion rate (% tasks fully solved).
- Checkpoint-based score / partial credit (percentage of points earned across checkpoints), intended to reflect progress on long-horizon tasks.
- (Implied) breakdowns by task type / difficulty; and error analyses about failures on longer/harder tasks.

## What are the main results?

- The best tested baseline agent/model completes about 30% of tasks fully (they cite 30.3% for Gemini 2.5 Pro in the HTML version).
- Checkpoint/partial-credit score is higher (they cite 39.3%), suggesting agents often make partial progress without finishing.
- Overall conclusion: agents can autonomously solve many simpler professional tasks, but struggle on difficult long-horizon tasks.

## How is this similar to GALILEO?

- Both are *benchmark + protocol* papers focused on measuring failures that matter in realistic agent deployments, with an emphasis on reproducibility and diagnostic signal beyond a single aggregate number.
- Both use richer evaluation than “one-shot accuracy”:
  - TheAgentCompany: checkpoint-based partial credit.
  - GALILEO: multi-round survival / turn-of-failure (TOF) and recovery conditional on initial correctness.

## How is this different from GALILEO?

- TheAgentCompany is an *environment + task-execution* benchmark for tool-using agents in a workplace setting (web/apps/terminal/coworkers).
- GALILEO is a *multi-turn belief-consistency / persuasion-robustness* benchmark on ground-truth QA-style tasks, focused on whether models maintain correct answers under repeated persona pressure and whether they recover after flipping.
- TheAgentCompany primarily stresses planning, tool-use, and end-to-end execution; GALILEO stresses conversational dynamics, sycophancy/persuasion, and correctness preservation over rounds.

## Where GALILEO is stronger / cleaner (if true)

- Cleaner causal isolation of multi-turn conversational drift: conditioning on round-0 correctness + neutral re-asking control to separate “pressure-induced” changes from baseline instability.
- More directly tied to *belief/answer consistency under social pressure*, which is under-measured in typical agent-task benchmarks.

## Where GALILEO is weaker / needs to improve

- Less coverage of realistic workplace toolchains (browsers, internal apps, coding + execution, collaboration tooling) and long-horizon task completion.
- No environment state evaluators/checkpoints for complex workflows; thus less direct signal on “can the agent do the job?” end-to-end.

## Action items for GALILEO (experiments / method / writing)

- [ ] Related work positioning: cite TheAgentCompany as complementary to GALILEO (execution/workplace capability vs multi-turn persuasion robustness).
- [ ] Consider importing one idea: checkpoint-style partial credit for multi-round robustness (e.g., credit for maintaining correctness through round k, or for partial recovery behaviors).
- [ ] Contrast section: explicitly argue why tool-use/workplace benchmarks do not measure belief-consistency failure modes (pressure-induced flips) and vice versa.

## Quotes / details to potentially cite

- Abstract: benchmark evaluates agents “by browsing the Web, writing code, running programs, and communicating with other coworkers.”
- They report: “175 diverse, realistic and professional tasks” (from the introduction figure caption in the HTML version).
- They report: best agent completes “30.3%” of tasks fully; “39.3%” on partial-credit metric (HTML introduction).
