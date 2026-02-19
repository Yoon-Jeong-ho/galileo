# CORE-Bench: Fostering the Credibility of Published Research Through a Computational Reproducibility Agent Benchmark

- Year: 2024
- Venue: arXiv
- Authors: Zachary S. Siegel, Sayash Kapoor, Nitya Nadgir, Benedikt Stroebl, Arvind Narayanan
- URL: https://arxiv.org/abs/2409.11363
- BibTeX key (if we add it): Siegel2024COREBench
- Tags: agents, computational reproducibility, benchmark, evaluation harness

## One-sentence takeaway

CORE-Bench is a 270-task benchmark (from 90 CodeOcean papers) that evaluates whether AI agents can successfully reproduce computational results end-to-end and answer grounded questions about reproduced outputs, with baseline accuracies topping out around 21% on the hardest tasks.

## What problem does it solve?

- Lack of realistic, end-to-end benchmarks for *agentic* scientific work that directly maps to a valuable real-world activity: **computational reproducibility** (install deps, run code, find the right outputs, and report results).
- Makes evaluation more practical via a **parallelizable harness** (avoiding “run tasks sequentially for weeks” style evaluation).

## What is the core method / protocol?

- Benchmark built from **CodeOcean capsules** (chosen because they are more likely to be runnable/reproducible than arbitrary GitHub repos), spanning **90 papers** across **computer science, social science, and medicine**.
- For each paper: **3 tasks / difficulty levels** (270 tasks total). The paper frames difficulty by what reproduction information is provided to the agent (e.g., access to outputs vs Dockerfile vs README-only).
- Task format: agent must (i) set up environment, (ii) execute reproduction, then (iii) answer one or more questions about outputs (numbers, labels, figure-reading; includes **language-only** and **vision-language** tasks).
- Scoring: a task is correct only if the agent answers **all** associated questions correctly (intended to reduce “guessing” success).
- Evaluation system: tasks run in isolated VMs and are executed **in parallel** to reduce wall-clock time and prevent benchmark tampering.

## What are the key metrics?

- **Task accuracy** (all-questions-correct per task), reported by task difficulty level.
- Comparisons across agents (AutoGPT vs task-specialized CORE-Agent) and across base models (GPT-4o vs GPT-4o-mini).

## What are the main results?

- Specialized agent (CORE-Agent) substantially outperforms a generic baseline on easier tasks (reported up to ~60% on easiest level).
- Performance drops sharply with task difficulty; the best setting reported achieves **~21% accuracy on the hardest level**, indicating substantial headroom.
- The evaluation harness reduces evaluation time dramatically (paper claims days/weeks → hours via parallelization).

## How is this similar to GALILEO?

- If GALILEO involves evaluating complex, multi-step agent behavior, CORE-Bench is a strong example of:
  - **end-to-end task definition** grounded in a real workflow,
  - **hard-to-game** evaluation (grounded questions; all-or-nothing correctness),
  - infrastructure emphasis (parallel evaluation, isolation).

## How is this different from GALILEO?

- Domain focus is computational reproducibility (running code + retrieving outputs), not GALILEO’s core domain/task framing.
- Tasks are derived from **CodeOcean** capsules under curated constraints (e.g., runtimes, capsule size), which may be less “in-the-wild” than arbitrary real repos.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO targets more open-world settings, it may better capture distribution shift beyond curated capsules.
- If GALILEO includes richer trajectory-level metrics (survival/recovery, partial credit, calibration), it may provide more diagnostic signal than all-or-nothing accuracy.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently lacks a robust harness for **parallel, isolated, reproducible** evaluation, CORE-Bench’s infrastructure is a useful reference point.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider borrowing the **“all questions must be correct”** style scoring for subtests where guessing is plausible.
- [ ] If we run agent evaluations that are slow/fragile, consider adopting the “**parallel isolated runners**” framing in the evaluation section (even if implementation differs).
- [ ] Add a short related-work paragraph positioning GALILEO relative to “agent benchmarks grounded in scientific workflows,” with CORE-Bench as an example.

## Quotes / details to potentially cite

- CORE-Bench evaluates agents on computational reproducibility: reproduce results from a paper given its repository (install deps, run code, then answer questions about outputs).
- Benchmark scale: **270 tasks from 90 papers** across **three disciplines**; includes language-only and vision-language tasks.
- Reported headroom: best accuracy around **21% on the hardest level**.
