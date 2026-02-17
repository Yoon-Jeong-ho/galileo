# ToolACE-MT: Non-Autoregressive Generation for Agentic Multi-Turn Interaction

- Year: 2026
- Venue: ICLR 2026 (per arXiv page)
- Authors: Xingshan Zeng, Weiwen Liu, Lingzhi Wang, Liangyou Li, Fei Mi, Yasheng Wang, Lifeng Shang, Xin Jiang, Qun Liu
- URL: https://arxiv.org/abs/2508.12685
- BibTeX key (if we add it): toolace_mt_2026
- Tags: agents, tool-use, multi-turn, data-generation, non-autoregressive, iterative-refinement

## One-sentence takeaway

ToolACE-MT proposes a non-autoregressive, turn-level “mask-and-fill” iterative refinement pipeline to generate full multi-turn tool-use trajectories more efficiently (and with more global consistency control) than expensive multi-agent autoregressive simulations.

## What problem does it solve?

- Generating *high-quality* multi-turn, multi-step tool-use interaction data is expensive with standard multi-agent simulation (MAS), because it requires long back-and-forth autoregressive generation.
- MAS also makes it hard to explicitly control dialogue length/complexity and to enforce global consistency/solvability because each turn is generated myopically.

## What is the core method / protocol?

- **Non-autoregressive trajectory generation at the conversation level** (inspired by non-autoregressive translation + masked diffusion / mask-predict style refinement).
- Pipeline with **three stages**:
  1) **Coarse-grained initialization**: create a *structurally complete but semantically coarse* dialogue skeleton (task + action trajectory).
  2) **Iterative refinement**: repeatedly apply **mask-and-fill** operations to inject realism/complexity and improve “reasonability” (their term) while keeping the overall structure.
  3) **Offline verification**: rule-based + model-based checks to filter/validate correctness/coherence.
- Key claimed benefit: having access to the *global* trajectory lets the generator optimize structure/consistency across turns and steps, rather than relying on the assistant model’s local autoregressive choices.

## What are the key metrics?

- Downstream performance of models trained with ToolACE-MT-generated data, evaluated on **agentic multi-turn benchmarks** (mentioned):
  - BFCL-v3 (Berkeley Function Calling Leaderboard)
  - \tau-Bench
  - ACEBench
- (Need to check the paper for exact reported metrics per benchmark, e.g., end-to-end accuracy / success rate, tool-call correctness, etc.)

## What are the main results?

- Claim: models trained on ToolACE-MT-generated data **outperform** models trained on data generated via autoregressive multi-agent simulation, across several agentic multi-turn benchmarks.
- Additional claims: improved **efficiency**, **complexity control**, and **generalization across backbones**.

## How is this similar to GALILEO?

- Shares the framing that multi-turn interactions require **global structure/trajectory-level reasoning** and that purely autoregressive turn-by-turn behavior can degrade consistency/solvability.
- Emphasizes evaluation over **multi-turn, multi-step agent/tool trajectories**, which is often where “drift”/failures appear.

## How is this different from GALILEO?

- ToolACE-MT is primarily about **data generation** for training tool-use agents (non-autoregressive synthesis + verification), not about *measuring* or *characterizing* multi-turn robustness failures per se.
- Focus is on **efficiency + controllability** of synthetic trajectory generation, whereas GALILEO-related work often focuses on **robustness/stability metrics** under pressure/perturbations.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO provides *explicit robustness metrics / stress tests* for multi-turn behavior, that’s conceptually cleaner for “robustness measurement” than conflating it with data-generation improvements.

## Where GALILEO is weaker / needs to improve

- If GALILEO needs scalable multi-turn datasets with *controlled difficulty and known ground-truth tool trajectories*, ToolACE-MT-style non-autoregressive skeleton+refinement could help generate such controlled stress-test suites.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider citing ToolACE-MT as evidence that **turn-level non-autoregressive refinement** is an emerging design pattern for *controlling* multi-turn trajectories (contrast with purely autoregressive MAS).
- [ ] If we need controllable multi-turn stress tests: adapt the **(skeleton → iterative refinement → verification)** pattern to generate *paired* trajectories with targeted perturbations (e.g., add misleading user turns, tool failures, or constraint changes).

## Quotes / details to potentially cite

- “ToolACE-MT generates full conversational trajectories through three stages: coarse-grained initialization, iterative refinement, and offline verification.”
- “Existing simulation-based data generation methods … rely heavily on costly autoregressive interactions between multiple LLM agents … compromising the practical efficiency …”
- “Unlike traditional autoregressive multi-agent simulations (MAS), ToolACE-MT generates full conversational trajectories through a non-autoregressive pipeline …”
