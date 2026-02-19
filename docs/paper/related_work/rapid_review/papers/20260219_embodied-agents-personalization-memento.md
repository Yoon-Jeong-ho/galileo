# Embodied Agents Meet Personalization: Investigating Challenges and Solutions Through the Lens of Memory Utilization

- Year: 2025
- Venue: ICLR 2026 (arXiv)
- Authors: Taeyoon Kwon; Dongwook Choi; Hyojun Kim; Sunghwan Kim; Seungjun Moon; Beong-woo Kwak; Kuan-Hao Huang; Jinyoung Yeo
- URL: https://arxiv.org/abs/2505.16348
- BibTeX key (if we add it): kwon2025memento
- Tags: personalization, memory, embodied-agents, retrieval, evaluation

## One-sentence takeaway

Introduces **MEMENTO**, a two-stage benchmark for personalized embodied assistance via memory (object semantics + user routines), showing current LLM agents recall simple facts but fail to plan with sequential routine memories due to overload and multi-memory coordination failures.

## What problem does it solve?

- Conventional embodied-agent benchmarks mostly test single-turn instruction following with static goals, which under-measures what’s needed for *personalized assistance*.
- Personalized assistance requires using user-specific knowledge from prior interactions (episodic memory) to interpret underspecified requests ("my favorite cup") and to follow routines ("my breakfast routine").

## What is the core method / protocol?

- Proposes **MEMENTO**, an end-to-end evaluation framework with:
  - **Object semantics** tasks: identify objects via user-specific meaning (favorite item, personal label).
  - **User patterns** tasks: recall and use *sequential* routines / object-location sequences in planning.
  - **Single-memory** tasks vs **joint-memory** tasks (requires combining multiple memories).
- Empirical analysis of LLM-powered embodied agents under different memory retrieval conditions (e.g., varying number of retrieved memories / top-k).
- Explores architectural mitigations; key proposal is a **hierarchical knowledge-graph user-profile memory** that separates personalized profile knowledge from other episodic memories.

## What are the key metrics?

- Task success / completion on MEMENTO tasks (single-memory and joint-memory).
- Sensitivity to retrieval size (performance vs top-k retrieved memories) to diagnose **information overload**.
- Error modes around combining memories (failures on joint-memory tasks) to diagnose **coordination failures**.

## What are the main results?

- Agents can often use memory for **simple object semantics**.
- Agents struggle substantially on **user pattern / routine** tasks that require sequential reasoning and planning with memory.
- Two highlighted bottlenecks:
  - **Information overload**: more retrieved memories add noise and degrade performance.
  - **Coordination failures**: agents fail to jointly use multiple relevant memories even in simple cases.
- A hierarchical KG-based user-profile memory module yields **substantial improvements** on both single and joint-memory tasks (per abstract).

## How is this similar to GALILEO?

- Both care about *multi-step behavior* where the agent must incorporate non-parametric context (memory/history) rather than only follow a single instruction.
- Both highlight that retrieval + reasoning/planning is a weak link: more context can make models worse (overload / distraction).

## How is this different from GALILEO?

- Focuses on **embodied** rearrangement-style tasks and personalization (object semantics, household routines), not primarily on conversational robustness/sycophancy-style pressure.
- Emphasizes memory *utilization* and architecture (episodic vs profile memory separation) more than adversarial dialogue protocols.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO uses a controlled multi-turn protocol with explicit perturbations/pressure, it may offer cleaner causal diagnostics of *why* an agent drifts (vs. broad embodied failure modes).

## Where GALILEO is weaker / needs to improve

- Could be missing explicit evaluation of **personalization memory** (user-specific semantics/routines) and the overload/coordination stress tests from MEMENTO.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a short related-work paragraph: personalization + memory utilization in embodied agents; cite MEMENTO as evidence of overload + multi-memory coordination as key failure modes.
- [ ] Consider borrowing their diagnostic axes: (a) single-memory vs joint-memory; (b) sequential routine memory vs isolated facts.
- [ ] If GALILEO has retrieval, add an ablation on **top-k** retrieved items to test for information-overload degradation.

## Quotes / details to potentially cite

- “We construct Memento, an end-to-end two-stage evaluation framework comprising single-memory and joint-memory tasks.”
- “Two critical bottlenecks: information overload and coordination failures when handling multiple memories.”
- “Agents can recall simple object semantics but struggle to apply sequential user patterns to planning.”
