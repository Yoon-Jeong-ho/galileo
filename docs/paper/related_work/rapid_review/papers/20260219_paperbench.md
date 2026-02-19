# PaperBench: Evaluating AI’s Ability to Replicate AI Research

- Year: 2025
- Venue: arXiv (benchmark paper; targets ICML 2024 Spotlight/Oral papers)
- Authors: Giulio Starace, Oliver Jaffe, Dane Sherburn, James Aung, Jun Shern Chan, Leon Maksin, Rachel Dias, Evan Mays, Benjamin Kinsella, Wyatt Thompson, Johannes Heidecke, Amelia Glaese, Tejal Patwardhan
- URL: https://arxiv.org/abs/2504.01848 (v3)
- BibTeX key (if we add it): PaperBench2025Starace
- Tags: agents, long-horizon, benchmark, reproducibility, llm-judge

## One-sentence takeaway

PaperBench evaluates whether autonomous coding agents can replicate the empirical results of recent top-tier ML papers from scratch using author-approved, hierarchical rubrics and (optionally) an LLM judge.

## What problem does it solve?

- We lack realistic, *long-horizon* evaluations of AI agents’ ability to do end-to-end ML R&D work (read a paper, implement experiments, run them, troubleshoot) rather than short coding tasks or tasks with provided repos.
- Human grading of “did it replicate the paper?” is extremely expensive; PaperBench proposes rubric-based decomposition + an automated judge to scale evaluation.

## What is the core method / protocol?

- Dataset: 20 ICML 2024 Spotlight/Oral papers across multiple topics.
- For each paper, they build a **hierarchical rubric** (tree of requirements) co-developed with an original paper author.
  - Total: **8,316 leaf requirements** across the 20 papers.
  - Leaf nodes are binary pass/fail; internal nodes aggregate via manual weights.
- Candidate agent receives: paper + an addendum of author clarifications.
- Candidate must produce a *fresh repo from scratch* with a `reproduce.sh` entrypoint.
- **Reproduction phase**: copy submission into a fresh VM and run `reproduce.sh` (12h cap in their experiments) to generate logs/artifacts.
- **Grading phase**: an LLM-based judge (“SimpleJudge”) grades leaf nodes using paper markdown + rubric JSON + selected submission files.
  - They also introduce **JudgeEval**: a benchmark to evaluate judge accuracy vs human gold labels.
- Variant: **PaperBench Code-Dev** skips reproduction and grades only “code development” requirements (cheaper, noisier).

## What are the key metrics?

- **Replication Score**: weighted average of satisfied rubric requirements (root score), averaged across papers.
- Judge quality on JudgeEval: binary classification metrics (F1, etc.) against human-labeled leaf nodes.
- They also compare to a **human baseline** (ML PhDs) on a subset, under time limits.

## What are the main results?

- Best-performing tested agent in their main setup: **Claude 3.5 Sonnet (New)** with open-source scaffolding, **21.0%** average replication score.
- Human baseline (ML PhDs) on a 3-paper subset after 48 hours (best@3): **41.4%** vs **26.6%** for o1 on the same subset.
- Judge: their o3-mini-based judge with scaffolding achieves **F1 ≈ 0.83** on JudgeEval (per paper).
- Qualitative: many models terminate early or fail to execute long-horizon plans; agent scaffolding matters (IterativeAgent boosts o1/o3-mini scores; can hurt other models).

## How is this similar to GALILEO?

- Both are **evaluation frameworks** focused on moving beyond one-shot accuracy to assess capability under more realistic interaction loops.
- Both emphasize **auditable protocols** and decomposition/standardization to enable reproducible comparisons.
- The “agent scaffolding matters” finding is conceptually aligned with GALILEO’s emphasis on protocol design (controls, multi-round structure) affecting observed behavior.

## How is this different from GALILEO?

- PaperBench targets **autonomous ML engineering/research replication** (coding + running experiments), not multi-turn conversational **belief consistency under pressure**.
- PaperBench’s primary unit is a **software repo submission** graded via rubrics/judges; GALILEO’s unit is **multi-round dialogue trajectories** with survival/TOF/recovery metrics.
- PaperBench explicitly grapples with **LLM-judge reliability** and cost at very large scale; GALILEO’s scoring is typically grounded in task ground truth (math/QA/MCQA/OpenQA) with control arms to isolate drift.

## Where GALILEO is stronger / cleaner (if true)

- Clearer **ground-truth evaluation** (objective answer checking) and explicit control arm (Neutral Re-asking Control) to separate persona pressure from generic multi-turn drift.
- Much cheaper/faster to run at scale compared to “replicate ICML paper from scratch” evaluations.

## Where GALILEO is weaker / needs to improve

- PaperBench highlights a complementary axis: **end-to-end autonomy** in long-horizon tool use (code + execution). If GALILEO wants to claim broader “agent robustness” relevance, it may need at least a bridge discussion/ablation connecting conversational drift robustness to long-horizon agent competence.
- PaperBench’s rubric approach suggests ways to create **graded partial-credit** evaluations; GALILEO is mostly binary per-round correctness (though dynamics metrics enrich it).

## Action items for GALILEO (experiments / method / writing)

- [ ] Related-work positioning: cite PaperBench as evidence that (i) long-horizon autonomy benchmarks exist and (ii) **scaffolding/protocol choices strongly modulate results**, motivating GALILEO’s careful control design.
- [ ] Consider a “rubric-like” decomposition for a subset of GALILEO failures (e.g., classify failure modes per round: deference/authority, logical trap, denial, etc.) to yield partial-credit diagnostics beyond correctness.
- [ ] Add a short paragraph in discussion: PaperBench evaluates *agentic coding autonomy*; GALILEO evaluates *multi-turn truthfulness/belief consistency under social pressure*; both are needed for a fuller autonomy-risk picture.

## Quotes / details to potentially cite

- “Agents must replicate 20 ICML 2024 Spotlight and Oral papers from scratch…”
- “PaperBench contains 8,316 individually gradable tasks.”
- “The best-performing tested agent … achieves an average replication score of 21.0%.”
- JudgeEval / SimpleJudge framing: rubric-based grading with an LLM judge reaching F1 around 0.83 on their auxiliary judge benchmark.
