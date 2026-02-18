# PAG: Multi-Turn Reinforced LLM Self-Correction with Policy as Generative Verifier

- Year: 2025
- Venue: arXiv
- Authors: Yuhua Jiang; Yuwen Xiong; Yufeng Yuan; Chao Xin; Wenyuan Xu; Yu Yue; Qianchuan Zhao; Lin Yan
- URL: https://arxiv.org/abs/2506.10406
- BibTeX key (if we add it): jiang2025pag
- Tags: self-correction, verifier, multi-turn, reinforcement-learning

## One-sentence takeaway

PAG trains a single LLM (via multi-turn RL) to alternate between *verifying* its own answer and *revising* only when the generative verification step flags an error, improving both solution quality and self-verification.

## What problem does it solve?

- LLMs can reason well but are unreliable at *knowing when they are wrong* (self-verification).
- Many self-correction approaches either:
  - use a separate verifier model, or
  - use multi-stage pipelines, or
  - force a second attempt regardless of confidence (wasting turns / risking degeneration).

## What is the core method / protocol?

- “Policy as Generative Verifier (PAG)” in a unified multi-turn RL setup.
- The model alternates roles across turns:
  - Turn A (policy): produce an answer.
  - Turn B (verifier): generate a verification/critique signal.
  - If verifier detects an error → revise; otherwise stop (selective revision / verify-then-revise).
- Claimed benefit: selective revision reduces unnecessary rewrites and helps avoid “model collapse” during self-correction training.

## What are the key metrics?

- Direct-generation accuracy on reasoning benchmarks.
- Self-correction success rate / accuracy after verify-then-revise.
- Verifier quality: self-verification performance compared to self-consistency (details not captured from abstract).

## What are the main results?

- As a *policy*: improves both first-pass answers and self-correction accuracy across diverse reasoning benchmarks.
- As a *verifier*: generative self-verification outperforms self-consistency (per abstract).

## How is this similar to GALILEO?

- Shares the “multi-turn protocol matters” framing: performance depends on *when* to revise vs persist.
- Treats instability/correction as a sequential decision process (stop vs continue / revise vs keep).
- Emphasizes evaluation beyond single-shot accuracy (self-verification and correction behavior).

## How is this different from GALILEO?

- Focuses on *training* a self-correcting model with RL (policy+verifier in one model), rather than primarily defining an evaluation protocol for multi-turn robustness/drift (if GALILEO is eval-centric).
- Targets general reasoning correctness and verification; not specifically social influence / drift / long-horizon goal stability.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO is about measuring multi-turn instability/drift: it can be model-agnostic and cheaper than RL training, and may provide cleaner causal attribution via controls.

## Where GALILEO is weaker / needs to improve

- GALILEO may lack an explicit *learned* stop/revise controller; PAG suggests “selective revision” is important to prevent unnecessary turn extensions and degeneration.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding a GALILEO baseline/protocol variant: **selective intervention** (only apply a correction/revision step when an internal/external verifier crosses a threshold), and report turn savings vs performance.
- [ ] In writing, explicitly contrast “always produce N samples / self-consistency” vs “verify-then-revise” as different multi-turn control policies.

## Quotes / details to potentially cite

- “Policy as Generative Verifier (PAG) … alternating between policy and verifier roles within a unified multi-turn reinforcement learning (RL) paradigm.”
- “PAG introduces a selective revision mechanism: the model revises its answer only when its own generative verification step detects an error.”
