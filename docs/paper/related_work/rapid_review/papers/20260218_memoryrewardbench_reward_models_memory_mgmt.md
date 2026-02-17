# MemoryRewardBench: Benchmarking Reward Models for Long-Term Memory Management in Large Language Models

- Year: 2026
- Venue: arXiv
- Authors: Zecheng Tang, Baibei Ji, Ruoxi Sun, Haitian Wang, Wangjie You, Yijun Zhang, Wenpeng Zhu, Ji Qi, Juntao Li, Min Zhang
- URL: https://arxiv.org/abs/2601.11969
- BibTeX key (if we add it): memoryrewardbench2026tang
- Tags: long-context, memory, reward-model, benchmark, multi-turn

## One-sentence takeaway

MemoryRewardBench benchmarks *reward models* (not LLMs) on judging the quality of long-term memory-management trajectories (8K–128K context) across reasoning, multi-turn dialogue, and long-form generation.

## What problem does it solve?

- Many long-context / segmented-processing systems rely on intermediate “memories” (summaries, state updates); we need a reliable automatic evaluator to score these intermediate memories.
- Prior memory benchmarks mostly evaluate LLM task performance directly; this work asks whether reward models can supervise/evaluate the *memory management process* itself.

## What is the core method / protocol?

- Construct a benchmark (10 settings) spanning:
  - long-context comprehension / reasoning
  - multi-turn dialogue
  - long-form generation
- Each test instance presents an RM with:
  - the original context (8K–128K tokens)
  - two candidate memory-management trajectories
  - the associated final outcomes
- RM chooses the better trajectory and provides a justification.
- Two criteria types:
  - Outcome-based: prefer trajectories that lead to correct outcomes.
  - Process-based: both outcomes are correct; prefer the trajectory with better memory updates (accurate, concise, coherent), i.e., decouple memory-quality from outcome correctness.

## What are the key metrics?

- Pairwise preference accuracy / win-rate of the RM when selecting the superior trajectory under the benchmark’s criteria (outcome-based vs process-based).

## What are the main results?

- Evaluations on 13 “cutting-edge” RMs (3 proprietary, 10 open-source) suggest:
  - open vs proprietary gaps are shrinking
  - newer-generation models tend to outperform older ones regardless of parameter count
- The paper emphasizes limitations of current RMs for reliably evaluating memory management across diverse settings.

## How is this similar to GALILEO?

- If GALILEO uses LLM-as-judge or preference modeling for multi-turn behaviors, this is directly adjacent: it treats evaluation as a learned judge and focuses on multi-turn / long-context dependencies.
- The *process-based* criterion resembles evaluating “trajectory quality” rather than only final-answer correctness.

## How is this different from GALILEO?

- Target is specifically reward models judging intermediate memory updates (segmented processing), rather than end-to-end dialogue quality or task success alone.
- The benchmark protocol is pairwise comparison of memory-management trajectories with explicit decoupling of process vs outcome.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO has clearer task definitions tied to downstream user outcomes, it may be more actionable than a judge-of-memory benchmark.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently evaluates mostly final responses, it may miss failure modes where memory updates are poor but final outcomes look fine.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a “process-based” evaluation mode: hold final outcome constant (both correct) and assess whether intermediate state/memory updates are faithful, minimal, and coherent.
- [ ] Consider a pairwise rubric + judge setup for comparing two dialogue trajectories (or two internal memory traces) rather than scoring single outputs.
- [ ] Add related-work paragraph positioning: most memory benchmarks evaluate LLMs; MemoryRewardBench evaluates *reward models* for supervising memory management.

## Quotes / details to potentially cite

- “MemoryRewardBench … covers both long-context comprehension and long-form generation tasks … 10 distinct settings … context length ranging from 8K to 128K tokens.”
- “For each evaluation, RM is provided with the original context … two candidate memory management trajectories, and their respective outcomes. The RM’s task is to select the superior sample … while also providing a justifying explanation.”
- Criteria split: outcome-based vs process-based (both outcomes correct; prefer better memory updates).