# MLLM-CTBench: A Benchmark for Continual Instruction Tuning with Reasoning Process Diagnosis

- Year: 2025
- Venue: arXiv (under review)
- Authors: Haiyun Guo, Zhiyan Hou, Yandu Sun, Jinghan He, Yu Chen, Yuzhe Zhou, Yuheng Jia, Jinqiao Wang, Tat-Seng Chua
- URL: https://arxiv.org/abs/2508.08275
- BibTeX key (if we add it): mllmctbench2025guo
- Tags: continual-learning, instruction-tuning, multimodal, catastrophic-forgetting, evaluation, reasoning-traces, rft

## One-sentence takeaway

MLLM-CTBench proposes a protocol-consistent continual-instruction-tuning benchmark for multimodal LLMs that evaluates both final-answer accuracy and reasoning-process quality (via CoT traces) and finds RFT with explicit KL control (GRPO) retains prior tasks more stably than sequential SFT.

## What problem does it solve?

- Lack of rigorous, unified benchmarks for *continual instruction tuning* (CIT) of multimodal LLMs (MLLMs) with consistent protocols and diagnostics.
- Prior CIT benchmarks focus on answer correctness, giving weak visibility into *why* catastrophic forgetting happens.

## What is the core method / protocol?

- Benchmark: 7 tasks across 6 domains for continual instruction tuning of MLLMs.
- Evaluation is **multi-dimensional**:
  - final-answer accuracy
  - **process-level reasoning quality** using Chain-of-Thought traces as an observable diagnostic signal.
- Runs multiple continual-learning algorithms (8 methods, grouped into 4 families) under a unified protocol and across different task orders.
- Extends from sequential **SFT** to **RFT**, studying GRPO (on-policy RL) and the role of **explicit KL-divergence control** to a prior policy for stability/retention.

## What are the key metrics?

- Post-task vs final performance per task (catastrophic forgetting), reported for:
  - answer-level correctness
  - CoT/process-quality scores (incl. dimensions like visual grounding fidelity, logical coherence, domain-knowledge retention).

## What are the main results?

- Process-level reasoning quality degrades more slowly than final-answer accuracy under continual tuning (i.e., it can be “more resilient” to forgetting).
- Forgetting is primarily driven by loss of **domain knowledge** (per their diagnostic breakdown).
- Larger/stronger baseline models show better resistance to forgetting.
- **RFT (GRPO) with KL control** yields more stable cross-task retention than SFT; removing KL control can amplify forgetting (even if it can help new-task gains).

## How is this similar to GALILEO?

- Shares the theme of **robustness under sequential change** (here: task sequence and continual updates; in GALILEO: robustness under evolving multi-turn interaction/pressure).
- Emphasizes **protocol-consistent evaluation** and the value of measuring *latent/process signals* rather than only final answers.

## How is this different from GALILEO?

- Not about user-induced multi-turn manipulation, sycophancy, or conversational drift; it is about **continual post-training** (model update sequences).
- Focuses on multimodal tasks and catastrophic forgetting, not interaction-time robustness.

## Where GALILEO is stronger / cleaner (if true)

- GALILEO can position itself as measuring *in-context, interaction-time* degradation (pressure/drift) rather than *training-time* forgetting.
- GALILEO may have clearer causal knobs for “pressure” vs “evidence” vs “control prompts,” whereas CIT mixes data distribution + optimization effects.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently only scores final outputs, MLLM-CTBench supports the argument that **process-level diagnostics** (e.g., rationale quality, grounding checks, knowledge-retention proxies) are important and can show different failure dynamics.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add (or at least discuss) a **process-level diagnostic** alongside outcome metrics (e.g., rationale consistency/grounding/knowledge-retention proxies) and check whether it degrades slower/faster than final answers under multi-turn pressure.
- [ ] In related work, cite as an example of “answer-only evaluation can miss important dynamics” and as motivation for multi-dimensional scoring.
- [ ] Consider a “KL-controlled update” analogy: if GALILEO uses adaptation or self-refinement across turns, explicitly control divergence from an initial stance/policy to prevent runaway drift.

## Quotes / details to potentially cite

- They argue CoT traces can be used as an “observable signal to diagnose catastrophic forgetting beyond answer-only evaluation.”
- Reported finding: “Process-level reasoning quality is often more resilient to catastrophic forgetting than final-answer accuracy.”
- They study GRPO and note KL control improves stability/retention; removing KL control can amplify forgetting.

