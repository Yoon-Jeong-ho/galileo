# Existing LLMs Are Not Self-Consistent For Simple Tasks

- Year: 2025
- Venue: arXiv
- Authors: Zhenru Lin; Jiawen Tao; Yang Yuan; Andrew Chi-Chih Yao
- URL: https://arxiv.org/html/2506.18781v1
- BibTeX key (if we add it): lin2025selfconsistency
- Tags: self-consistency, inconsistency-metrics, evaluation, graph, energy-based-model, category-theory, relational-reasoning

## One-sentence takeaway

Even strong LLMs produce logically inconsistent sets of pairwise relations on simple ordered/relational tasks, and the paper proposes task-agnostic inconsistency scores plus partial automated “fixing” via graph/energy-based optimization.

## What problem does it solve?

- Defines and quantifies *self-consistency* as “no contradictions under composition/transitivity” in the model’s implied relational structure (e.g., temporal order, 2D spatial comparisons, multi-hop family-tree relations).
- Shows that inconsistency persists even for modern reasoning models, motivating self-consistency as a prerequisite for interpretability/trust.

## What is the core method / protocol?

- Treat the model’s outputs over all ordered pairs of objects as a relational graph/category; self-consistency corresponds to satisfying composition laws (e.g., if A<B and B<C then A<C).
- Define an **inconsistency score** as the minimum edit needed to reach a composition-consistent extension of a trusted context.
  - Special cases described:
    - **Edit-to-consistency** when no ground-truth context is given (remove/reverse minimum edges to make structure consistent).
    - **Error rate** when ground truth fully specifies relations (count edges contradicting truth).
- Evaluate across domains:
  - 1D temporal ordering
  - 2D spatial comparisons (east-west vs north-south)
  - multi-hop kinship reasoning
- Propose two automated mitigation approaches:
  - **Graph-based fixing** (enforcing acyclicity / consistency by editing edges).
  - **Energy-based (EBM) fixing** to obtain a globally consistent assignment (details beyond the truncated extract, but positioned as complementary to graph approach).

## What are the key metrics?

- Inconsistency score I(A;C): normalized minimal edits from model relation set A to a consistent set extending trusted context C.
- Reported as percentage inconsistency (lower is better) across datasets; also compares before/after “fixing”.

## What are the main results?

- No tested model is fully self-consistent on these “simple” relational tasks when scaling to 11–51 objects (combinatorial number of pairwise comparisons).
- Smaller models show high inconsistency; larger/reasoning models (e.g., DeepSeek-R1, GPT-o4-mini) improve but remain imperfect.
- Graph-based and EBM-based scores correlate and can be used as cross-checking signals.
- When errors are sparse, automated fixing can partially recover ground truth, but not a full solution.

## How is this similar to GALILEO?

- Frames reasoning quality as **global structural consistency** over many local predictions (pairwise relations), rather than single-answer accuracy.
- Suggests evaluation/diagnostics that align with “system-level” reasoning reliability (consistency under composition).

## How is this different from GALILEO?

- Focuses narrowly on binary relational composition (orders/kinship-like relations) and post-hoc fixing, rather than end-to-end task performance or broader agentic pipelines.
- Leans on a category-theoretic framing; GALILEO may not require that machinery.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO already models structured state with explicit constraints, it may enforce consistency *by construction* (versus post-hoc repair of free-form LLM outputs).

## Where GALILEO is weaker / needs to improve

- If GALILEO evaluations emphasize final answer correctness, it may miss “hidden” internal contradictions that this paper operationalizes.
- If GALILEO relies on many pairwise/judged comparisons, it may inherit the same inconsistency failure mode without global constraint enforcement.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a “self-consistency” diagnostic: sample N objects in a task, query pairwise relations, then measure transitivity / composition violations.
- [ ] Report inconsistency alongside accuracy (especially for multi-hop/graph tasks) to argue interpretability/trust.
- [ ] Consider a constrained decoding / global optimization layer (graph acyclicity / ILP / energy minimization) to enforce relational consistency in predicted structures.
- [ ] Include a discussion paragraph: self-consistency vs reality-alignment (internally coherent but wrong), and which GALILEO targets.

## Quotes / details to potentially cite

- Abstract claim: even SOTA models “are not fully self-consistent” on simple tasks like comparing points (1D/2D) or family-tree reasoning; proposes “inconsistency metrics” and “graph-based and energy-based” automated fixes.
- Motivation example: if a model asserts A<B and B<C yet also that C is between A and B, its reasoning is contradictory.
- Conceptual framing: self-consistency and reality-alignment as two prerequisites for interpretability; next-token prediction may lack “backward edges” to enforce consistency (their discussion).
