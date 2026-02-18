# A Hierarchical and Evolvable Benchmark for Fine-Grained Code Instruction Following with Multi-Turn Feedback

- Year: 2025
- Venue: arXiv (cs.SE)
- Authors: Guoliang Duan et al.
- URL: https://arxiv.org/abs/2507.00699
- BibTeX key (if we add it): Duan2025MultiCodeIF
- Tags: benchmark, code-generation, instruction-following, constraints, multi-turn, feedback

## One-sentence takeaway

MultiCodeIF is a constraint-taxonomy-driven, multi-turn-evolvable benchmark that shows current LLMs struggle badly with *hierarchical / implicit* constraints but can improve substantially when given structured iterative feedback.

## What problem does it solve?

- Existing code-generation benchmarks often emphasize functional correctness while under-measuring whether models follow **layered, diverse, non-functional** instructions typical of real development.
- Lack of a **fine-grained constraint taxonomy** and **multi-turn (feedback → revision)** evaluation that can quantify instruction adherence beyond pass/fail.

## What is the core method / protocol?

- Introduces **MultiCodeIF**:
  - A structured taxonomy: **9 categories, 27 constraint types** to label instruction constraints.
  - Tasks include multiple programming languages (reported: **14**).
  - “Hierarchical levels” of constraints (single-level vs multi-level) to test compositional adherence.
- Uses an automated synthesis/evolution pipeline **ConstraGen**:
  - Generates and “evolves” code tasks (reported total: **2,021** tasks).
  - Produces **multi-turn variants** via feedback-driven refinement rounds.
- Evaluation reports **constraint satisfaction** rather than only functional correctness.

## What are the key metrics?

- **Average constraint satisfaction** (overall; also by explicit vs implicit/abstract constraints).
- Breakdown by **hierarchical constraint level** (single vs multi-level).
- Multi-turn improvement over iterative refinement rounds (feedback-driven).

## What are the main results?

- Large model spread: top model reported **63.0%** average constraint satisfaction (Claude-3-7-Sonnet) vs **44.8%** for a smaller model (Qwen3-1.7B).
- Explicit constraints are easier than **implicit / abstract** constraints.
- Multi-level constraints are much harder: success drops from **54.5% (single-level)** to **18.8% (multi-level)**.
- Structured feedback enables large gains across rounds: average satisfaction improves from **63.0% → 83.4%** over **4** refinement rounds.

## How is this similar to GALILEO?

- Shares the **multi-turn** framing: model behavior is evaluated across **iterative feedback rounds**, not just a single response.
- Emphasizes that “compliance/adherence” failures often appear when requirements become **layered** (analogous to multi-turn pressure/constraints compounding).

## How is this different from GALILEO?

- Domain is **code instruction following** (constraints on generated code) rather than social-pressure belief drift / persuasion dynamics.
- Primary outcome is **constraint satisfaction**; does not focus on flip dynamics, recovery-after-failure, or pressure-vs-evidence separation.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO’s contribution is about multi-turn *robustness under pressure* and *trajectory dynamics* (flip/recovery), MultiCodeIF is largely orthogonal: it does not operationalize those temporal failure/recovery patterns.

## Where GALILEO is weaker / needs to improve

- MultiCodeIF demonstrates a clean, reusable approach to:
  - defining a **constraint taxonomy**, and
  - reporting adherence broken down by **constraint types** and **hierarchical complexity**.
  If GALILEO lacks a similarly explicit “pressure/constraint taxonomy,” this is a useful design precedent.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider introducing an explicit **taxonomy** of pressure operators / constraint types (explicit vs implicit, single vs multi-level) and report results stratified by these types.
- [ ] If GALILEO has interventions/mitigations, consider reporting “round-by-round improvement” under structured feedback to mirror the **iterative refinement** story (even if the domain differs).

## Quotes / details to potentially cite

- “Tasks with multiple hierarchical constraints significantly reduce model success rates, from 54.5% in single-level to just 18.8% in multi-level scenarios.”
- “Structured feedback enables progressive improvement: average constraint satisfaction rises from 63.0% to 83.4% over four iterative refinement rounds.”
- Dataset/code release: https://github.com/SYSUSELab/MultiCodeIF
