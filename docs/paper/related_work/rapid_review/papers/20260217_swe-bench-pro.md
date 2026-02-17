# SWE-Bench Pro: Can AI Agents Solve Long-Horizon Software Engineering Tasks?

- Year: 2025
- Venue: arXiv
- Authors: Xiang Deng; Jeff Da; Edwin Pan; Yannis Yiming He; Charles Ide; Kanak Garg; Niklas Lauffer; Andrew Park; Nitin Pasari; Chetan Rane; Karmini Sampath; Maya Krishnan; Srivatsa Kundurthy; Sean Hendryx; Zifan Wang; Vijay Bharadwaj; Jeff Holm; Raja Aluri; Chen Bo Calvin Zhang; Noah Jacobson; Bing Liu; Brad Kenstler
- URL: https://arxiv.org/abs/2509.16941
- BibTeX key (if we add it): deng2025swebenchpro (suggested)
- Tags: agents, long-horizon, software-engineering, benchmark, contamination

## One-sentence takeaway

SWE-Bench Pro is a harder, more contamination-resistant successor to SWE-Bench that targets *enterprise-like*, multi-file, large-patch issue-resolution tasks and finds current coding agents still <45% Pass@1 under a unified scaffold.

## What problem does it solve?

- Existing repo-level SWE benchmarks can be (i) **contaminated** by public GitHub training data and (ii) **too easy / too small-change**, underrepresenting the multi-file, large-diff work common in professional engineering.
- Need a benchmark that better reflects **long-horizon** software engineering tasks (hours–days for a human engineer), and that supports diagnosis of failure modes from agent trajectories.

## What is the core method / protocol?

- **Benchmark construction**: 1,865 issue-resolution problems from 41 actively maintained repositories spanning business apps, B2B services, and dev tools.
- **Partitioning for contamination resistance / generalization checks**:
  - Public set: 11 GPL-licensed repos (problems + results released).
  - Held-out set: 12 GPL-licensed repos (not publicly accessible; intended for future overfitting checks).
  - Commercial set: 18 proprietary repos from startups (codebases private; results released).
- **Complexity filtering**: excludes “trivial edits”; keeps tasks requiring nontrivial multi-file modifications.
- **Human-centered augmentation/verification** (three-stage, per intro): clarifies task context and “recovers” tests as robust verifiers to reduce false negatives while keeping solution flexibility.
- **Evaluation**: runs multiple coding models under a **unified agent scaffold**; analyzes/clusters **failure modes** from collected trajectories.

## What are the key metrics?

- Primary: **Pass@1** (issue resolved / tests pass).
- Task-complexity descriptors (reported in intro): average reference solution size (**~107 LOC** across **~4.1 files**); minimum change **≥10 LOC**; many tasks with **>100 LOC** changes.
- Diagnostic: qualitative/clustered **failure mode taxonomy** based on agent trajectories (details likely in later sections).

## What are the main results?

- Across widely used coding models (under one scaffold), performance on SWE-Bench Pro remains **below 45% Pass@1** (per abstract).
- Benchmark emphasizes long-horizon, multi-file edits; designed to be more realistic than SWE-Bench Verified where many tasks are tiny patches.

## How is this similar to GALILEO?

- Shares the core motivation of moving from “toy” evaluation toward **deployment-realistic, long-horizon** settings.
- Uses **trajectory-level analysis** (cluster failure modes from agent runs), which aligns with GALILEO’s emphasis on analyzing multi-turn dynamics rather than only end-state accuracy.
- Explicitly addresses a key evaluation pitfall: **contamination** (analogous to how GALILEO must worry about benchmark leakage / memorization artifacts when measuring robustness).

## How is this different from GALILEO?

- Domain: SWE-Bench Pro is **software engineering issue resolution**; GALILEO is focused on **multi-turn belief/answer dynamics under pressure** (drift vs revision, recovery, etc.).
- Outcome definition: SWE-Bench Pro is mostly **end-to-end pass/fail** (tests passing), whereas GALILEO foregrounds **time-to-failure, flip dynamics, recovery**, and pressure-vs-evidence controls.
- Pressure/intervention: SWE-Bench Pro is not primarily about **persuasion / social pressure** or belief drift; it is about agentic coding workflows.

## Where GALILEO is stronger / cleaner (if true)

- GALILEO can provide **cleaner causal controls** (pressure-only vs evidence-bearing updates) and **turn-level** metrics (survival/time-to-event, recovery trajectories) that are not naturally captured by a single Pass@1 score.

## Where GALILEO is weaker / needs to improve

- Benchmark realism: SWE-Bench Pro highlights how important it is to ensure tasks are **industrial / long-horizon** and **human-verified**; GALILEO should be careful that its tasks aren’t inadvertently “short-horizon” or overly synthetic.
- Contamination resistance: their licensing + held-out/private split is a concrete pattern GALILEO could emulate for robustness claims.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider a **held-out split** strategy explicitly designed for contamination/overfitting checks (e.g., sources or domains that stay private until final evaluation).
- [ ] Add a short related-work paragraph acknowledging the parallel in **benchmark realism + contamination resistance**, even though the domain differs.
- [ ] If we use trajectory analyses, cite SWE-Bench Pro as precedent for **failure-mode clustering** from agent traces.

## Quotes / details to potentially cite

- “SWE-Bench Pro … designed to mitigate data contamination … (1) selecting repositories distributed under strong copyleft licenses (GPL) … and (2) acquiring commercial codebases …” (intro)
- “On average, the reference solutions span **107.4 lines of code across 4.1 files**.” (intro)
- “… performance on SWE-Bench Pro remains **below 45% (Pass@1)**.” (abstract)
