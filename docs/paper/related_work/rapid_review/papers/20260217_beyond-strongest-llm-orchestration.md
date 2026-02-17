# Beyond the Strongest LLM: Multi-Turn Multi-Agent Orchestration vs. Single LLMs on Benchmarks

- Year: 2025
- Venue: arXiv
- Authors: Aaron Xuxiang Tian, Ruofan Zhang, Jiayao Tang, Young Min Cho, Xueqian Li, Qiang Yi, Ji Wang, Zhunping Zhang, Danrui Qi, Zekun Li, Xingyu Xiang, Sharath Chandra Guntuku, Lyle Ungar, Tianyu Shi, Chi Wang
- URL: https://arxiv.org/abs/2509.23537
- BibTeX key (if we add it): tian2025beyond
- Tags: multi-turn, multi-agent, orchestration, evaluation, voting, consensus

## One-sentence takeaway

Multi-turn “orchestration” (iterative propose+vote among multiple frontier LLMs) can match/exceed the strongest single model on standard benchmarks, but design choices like revealing authorship or partial vote totals introduce self-voting and herding that can distort outcomes.

## What problem does it solve?

- When you have access to multiple LLMs, how should you combine them (over multiple turns) to reliably beat or at least match the best single model?
- What social/interaction design factors (authorship visibility, vote visibility) change convergence dynamics and failure modes?

## What is the core method / protocol?

- Multi-turn multi-agent orchestration loop:
  - Multiple LLM “agents” each propose answers.
  - Agents then cast votes (iteratively) until consensus / convergence.
- Two main experiments:
  1) Compare orchestration vs single-LLM baselines across tasks.
  2) Ablate orchestration design on GPQA-Diamond:
     - Whether agents can see who authored each candidate answer.
     - Whether agents can observe ongoing vote counts while voting.

## What are the key metrics?

- Task performance on:
  - GPQA-Diamond (hard science QA)
  - IFEval (instruction-following evaluation)
  - MuSR (multi-step reasoning)
- “Best-achievable orchestration performance” analysis (upper-bound style comparison from runs / configurations; details not fully visible from abstract).
- Qualitative/behavioral measures from ablations: self-voting rate, tie frequency, evidence of herding, convergence speed.

## What are the main results?

- Orchestration matches or exceeds the strongest single model, and consistently outperforms weaker single-model baselines.
- Revealing authorship increases self-voting and produces more ties.
- Showing ongoing vote totals amplifies herding; this speeds convergence but can cause premature consensus (locking onto suboptimal answers).

## How is this similar to GALILEO?

- Both care about *multi-turn dynamics* and how interaction protocols shape outcomes (not just single-turn accuracy).
- The ablations highlight protocol-induced artifacts (herding, self-voting), analogous to GALILEO’s interest in separating genuine capability changes from evaluation/protocol effects.

## How is this different from GALILEO?

- Focus is on *multi-agent aggregation* across multiple models, rather than within-agent drift/consistency under multi-turn interaction with a user or evolving context.
- Benchmarks emphasize final-task success, not “turn-of-failure” / instability trajectories / recovery-to-truth style measures.

## Where GALILEO is stronger / cleaner (if true)

- Likely stronger at diagnosing *when* and *why* a model changes its stance/answer over turns (trajectory-level metrics), not just whether a committee beats a baseline.
- Can explicitly control confounds between evidence-driven revision vs social/protocol pressure.

## Where GALILEO is weaker / needs to improve

- If GALILEO doesn’t include multi-agent orchestration baselines, it may miss a strong competing paradigm for robustness improvements (aggregation rather than improved single-agent stability).

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a “committee/orchestration” baseline (propose+vote) as a robustness comparator, then evaluate with GALILEO’s trajectory metrics (e.g., does orchestration reduce flip-flops or just mask them?).
- [ ] In GALILEO experimental design write-up, call out *interaction-design artifacts* (herding, self-voting) as a general risk when using multi-agent methods.
- [ ] If we ever include voting-style protocols, standardize: hide authorship and hide partial vote totals unless explicitly studying social dynamics.

## Quotes / details to potentially cite

- From the abstract (paraphrased for citation planning): revealing authorship increases self-voting and ties; showing ongoing votes amplifies herding, accelerating convergence but risking premature consensus.
