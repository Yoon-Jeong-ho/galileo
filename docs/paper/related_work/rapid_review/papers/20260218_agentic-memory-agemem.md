# Agentic Memory: Learning Unified Long-Term and Short-Term Memory Management for Large Language Model Agents

- Year: 2026
- Venue: arXiv
- Authors: Yi Yu; Liuyi Yao; Yuexiang Xie; Qingquan Tan; Jiaqi Feng; Yaliang Li; Libing Wu
- URL: https://arxiv.org/abs/2601.01885
- BibTeX key (if we add it): agemem2026agentic
- Tags: agents, memory, long-horizon, RL, GRPO, context-management

## One-sentence takeaway

AgeMem trains an LLM agent (via progressive RL + step-wise GRPO) to treat **both** long-term memory (add/update/delete) and short-term context control (retrieve/summarize/filter) as tool actions inside the policy, improving long-horizon benchmark performance and memory quality.

## What problem does it solve?

- Long-horizon agent tasks break down due to finite context windows.
- Prior work usually treats long-term memory (LTM) and short-term memory (STM) as separate modules (heuristics / fixed schedules / separate controllers), which limits end-to-end optimization and adaptability.

## What is the core method / protocol?

- **Unified tool interface**: memory operations are explicit tool actions the agent can invoke.
  - LTM tools: Add / Update / Delete.
  - STM tools: Retrieve / Summary / Filter (to compress or remove distractors/noise).
- **Three-stage progressive RL training**:
  1) Stage 1: casual interaction + learn to construct LTM (store salient info).
  2) Stage 2: reset short-term context; inject distractors; learn STM control (filter/summarize).
  3) Stage 3: solve a query requiring coordinated use of LTM retrieval + managed STM.
- **Step-wise GRPO**: terminal reward (task + memory + context rewards + penalties) is converted into a step-wise learning signal across the whole trajectory to address sparse/discontinuous rewards around memory ops.

## What are the key metrics?

- Task metrics vary by benchmark:
  - Success Rate (SR): ALFWorld, SciWorld, BabyAI
  - Progress Rate (PR): PDDL
  - LLM-as-a-judge score (J): HotpotQA
- Memory Quality (MQ): LLM-evaluated relevance of stored memories vs ground-truth supporting facts (on HotpotQA).
- Efficiency: average prompt token count (STM effectiveness) + tool-usage stats.

## What are the main results?

- Evaluated on **five** benchmarks: ALFWorld, SciWorld, PDDL, BabyAI, HotpotQA.
- Beats strong memory baselines (LangMem, A-Mem, Mem0, Mem0g) across two backbones (Qwen2.5-7B-Instruct, Qwen3-4B-Instruct).
- Reported averages:
  - 41.96% (Qwen2.5-7B) and 54.31% (Qwen3-4B); +4.82 to +8.57 pts over best baselines (Mem0/A-Mem).
  - RL contributes ~+8.5 pts over AgeMem-noRL.
- Memory quality: MQ up to **0.533** (Qwen2.5-7B) / **0.605** (Qwen3-4B) on HotpotQA.
- STM efficiency: replacing STM tools with RAG increases tokens; AgeMem reduces average prompt tokens by ~3.1% (Qwen2.5) / ~5.1% (Qwen3) vs an RAG variant.

## How is this similar to GALILEO?

- Both care about **multi-turn / long-horizon** agent behavior under limited context.
- Their Stage 2 “distractor” setting is adjacent to GALILEO’s interest in robustness under noisy / pressure-inducing interaction histories.
- Tool-based **filtering/summarization** resembles an explicit mechanism for resisting distraction-driven drift.

## How is this different from GALILEO?

- AgeMem’s primary contribution is **learning memory operations** (LTM+STM) via RL; it is not focused on belief drift / persuasion / sycophancy as an evaluation target.
- Uses benchmark suites (ALFWorld/SciWorld/PDDL/BabyAI/HotpotQA) rather than human-like multi-round pressure/sycophancy stress tests.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO isolates *multi-turn robustness failures* (e.g., belief drift, compliance under pressure) with targeted protocols, it may provide clearer causal attribution than broad task suites.

## Where GALILEO is weaker / needs to improve

- GALILEO may need a stronger story (or ablation) around **context/memory management** as a confounder/lever for multi-turn robustness.
- Might be missing explicit comparisons between:
  - static context vs learned summarization/filtering,
  - heuristic memory vs learned memory policies.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add an experiment axis: robustness metrics (belief stability / sycophancy resistance) under (a) naive full-history prompts vs (b) summarization vs (c) filtering vs (d) retrieval-augmented memory.
- [ ] Consider a “distractor phase” protocol (like their Stage 2) to stress-test whether robustness failures are driven by noisy intermediate context.
- [ ] In related work, cite AgeMem as evidence that **learned** STM control (filter/summarize) can outperform static RAG-style context expansion.

## Quotes / details to potentially cite

- “AgeMem exposes memory operations as tool-based actions, enabling the LLM agent to autonomously decide what and when to store, retrieve, update, summarize, or discard information.”
- Three-stage setup: Stage 1 constructs LTM; Stage 2 resets STM and adds distractors; Stage 3 requires coordinated retrieval + context control.
- Step-wise GRPO motivation: handles “sparse and discontinuous rewards induced by memory operations.”
