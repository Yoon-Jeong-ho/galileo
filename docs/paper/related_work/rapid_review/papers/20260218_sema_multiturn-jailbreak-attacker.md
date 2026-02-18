# SEMA: Simple yet Effective Learning for Multi-Turn Jailbreak Attacks

- Year: 2026
- Venue: ICLR 2026 (per arXiv comment)
- Authors: Mingqian Feng; Xiaodong Liu; Weiwei Yang; Jialin Song; Xuekai Zhu; Chenliang Xu; Jianfeng Gao
- URL: https://arxiv.org/abs/2602.06854
- BibTeX key (if we add it): sema2026multiturn
- Tags: multi-turn, jailbreak, red-teaming, attacker-training, RL, intent-drift

## One-sentence takeaway

SEMA trains an *open-loop* multi-turn jailbreak “attacker” via (1) self-generated non-refusal prompt SFT prefill and (2) RL with an intent-drift-aware reward, achieving much higher ASR than prior single- and multi-turn baselines.

## What problem does it solve?

- Multi-turn jailbreak attacks are closer to the real threat model for chatbots than single-turn prompts, but are harder due to:
  - exploration complexity (long-horizon search over multi-turn dialogs), and
  - intent drift (the attack conversation “wanders” away from the original harmful goal).
- Prior multi-turn approaches often depend on hand-written strategies/templates, victim feedback, or external data; they can be brittle or costly.

## What is the core method / protocol?

- **Two-stage attacker training** (no external strategy templates / external data, per authors’ framing):
  1) **Prefilling self-tuning (SFT prefill):** generate multi-turn adversarial prompts from a minimal prefix and fine-tune the attacker to produce non-refusal, well-structured multi-turn adversarial prompts. Goal: stabilize rollouts for later learning.
  2) **RL with intent-drift-aware reward:** train the attacker to keep producing valid adversarial multi-turn prompts while *maintaining the same harmful objective*.
- **Intent-drift-aware reward** combines three components (as described in the abstract):
  - intent alignment,
  - compliance risk,
  - level of detail.
- **Open-loop attack regime:** does not rely on *victim feedback* during attack generation; aims to reduce exploration complexity and unify single- and multi-turn settings.

## What are the key metrics?

- **Attack Success Rate (ASR)**, including **ASR@1** (success in one attempt / one generated attack instance).
- Evaluated across multiple datasets, victim models (closed + open), and jailbreak judges.

## What are the main results?

- Claims **SOTA multi-turn jailbreak performance**, beating:
  - single-turn baselines,
  - manually scripted/template multi-turn baselines,
  - ablations/alternatives (SFT-only and DPO variants).
- Example headline number (abstract): **80.1% ASR@1** averaged across three victim models on AdvBench, **+33.9%** over prior SOTA.

## How is this similar to GALILEO?

- Both study **multi-turn adversarial interaction dynamics** rather than one-shot prompting.
- Both highlight a key long-horizon failure mode akin to **drift**:
  - SEMA: *intent drift* away from the harmful objective.
  - GALILEO: *answer drift/flip* under repeated multi-turn pressure vs a neutral re-asking control.
- Both are useful as **stress tests**: SEMA for safety refusal robustness; GALILEO for ground-truth task robustness under social/persona pressure.

## How is this different from GALILEO?

- **Objective & threat model:**
  - SEMA trains an attacker to elicit *harmful content compliance* (jailbreak).
  - GALILEO measures *truthfulness/answer survival* on ground-truth tasks under persona pressure (with explicit neutral drift control).
- **Method type:**
  - SEMA is a *learning algorithm* (SFT + RL) to produce attacks.
  - GALILEO is primarily an *evaluation protocol/benchmarking framework* (survival/TOF/recovery metrics) rather than training an attacker policy.
- **Feedback usage:**
  - SEMA emphasizes an *open-loop* regime (no victim feedback) for attack generation.
  - GALILEO runs interactive multi-turn conversations and quantifies outcomes; it is not framed as open-loop policy learning.

## Where GALILEO is stronger / cleaner (if true)

- Explicitly separates **pressure-induced degradation** from **baseline multi-turn drift** via the **Neutral Re-asking Control**, which is a clean causal control that many jailbreak papers do not include.
- Uses **ground-truth tasks** and produces interpretable multi-turn metrics (Survival / TOF / Recovery) tied to correctness.

## Where GALILEO is weaker / needs to improve

- If GALILEO wants to position against *learned* multi-turn adversaries, it may need:
  - a stronger “attacker” component (possibly learned, not only fixed personas), or
  - an argument why persona pressure is the relevant adversary class for the targeted failure mode.

## Action items for GALILEO (experiments / method / writing)

- [ ] In related work, cite SEMA as evidence that **multi-turn adversaries benefit from explicitly controlling drift**; connect this to why GALILEO’s **neutral drift control** is methodologically important.
- [ ] Consider a discussion paragraph: **“intent drift” vs “answer drift”** as a shared long-horizon phenomenon across safety refusal and truthfulness settings.
- [ ] (Optional future) Explore whether a lightweight learned dialog policy (even open-loop) can serve as a stronger pressure baseline than hand-written personas.

## Quotes / details to potentially cite

- “Multi-turn jailbreaks capture the real threat model for safety-aligned chatbots…”
- “SEMA comprises two stages… Prefilling self-tuning… Reinforcement learning with intent-drift-aware reward…”
- “Our open-loop attack regime avoids dependence on victim feedback…”
- “SEMA performs an average 80.1% ASR@1… 33.9% over SOTA.”
