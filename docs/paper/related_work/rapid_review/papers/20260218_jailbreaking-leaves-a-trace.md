# Jailbreaking Leaves a Trace: Understanding and Detecting Jailbreak Attacks from Internal Representations of Large Language Models

- Year: 2026
- Venue: arXiv
- Authors: Sri Durga Sai Sowmya Kadali; Evangelos E. Papalexakis
- URL: https://arxiv.org/abs/2602.11495
- BibTeX key (if we add it): kadali2026jailbreaking
- Tags: jailbreak, safety, detection, interpretability, hidden-states, tensor-decomposition

## One-sentence takeaway

Jailbreak prompts induce consistent, layer-dependent latent patterns across multiple model families that can be used for lightweight detection and even inference-time mitigation via selectively bypassing “susceptible” layers.

## What problem does it solve?

- Prompt/output-level jailbreak defenses are brittle against adaptive attacks; the paper targets **representation-level signals** that distinguish jailbreak vs benign prompts and can be exploited for detection/mitigation.

## What is the core method / protocol?

- Extract internal representations layer-wise (they mention multi-head attention + layer outputs / hidden states) from multiple open models (GPT-J, LLaMA, Mistral, Mamba).
- Use a **tensor-based latent representation** approach (tensor decomposition) to identify structural differences between activations for benign vs jailbreak prompts.
- Turn these latent features into a **lightweight detector** (no fine-tuning; no auxiliary LLM judge).
- Use the same signals to rank “high-susceptibility” layers and mitigate at inference time by **bypassing selected layers** (demonstrated on an “abliterated” LLaMA-3.1-8B).

## What are the key metrics?

- Jailbreak detection performance (paper summary implies a detector; details not in the arXiv abstract snapshot).
- Mitigation success rate: % jailbreak attempts blocked.
- Benign retention: % benign prompts still behave acceptably / unchanged after mitigation.

## What are the main results?

- They report consistent latent-space patterns for jailbreak prompts across diverse architectures (Transformers + state-space model).
- Inference-time intervention: on an abliterated LLaMA-3.1-8B, selectively bypassing high-susceptibility layers **blocks 78%** of jailbreak attempts while preserving **94%** of benign behavior.
- Emphasis: no parameter updates; minimal overhead; architecture-agnostic direction.

## How is this similar to GALILEO?

- Shares the thesis that **internal signals** (not just surface text) can be diagnostic of problematic behavior.
- The “layer susceptibility” framing parallels GALILEO’s interest in *where* pressure/attack influence accumulates over turns, and suggests a monitorable internal risk signal.

## How is this different from GALILEO?

- Focus is **jailbreak detection/mitigation** (safety policy circumvention) rather than **belief drift / sycophancy vs evidence-driven revision** under social pressure.
- Uses **white-box access** to hidden activations and inference-time layer interventions; GALILEO’s core evaluation story is largely behavior/protocol/metrics and may need to remain usable in black-box settings.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO emphasizes protocol controls (pressure-only vs evidence) and trajectory metrics (flip timing, recovery, oscillation), it provides a clearer behavioral decomposition than a binary “jailbreak vs benign” split.
- GALILEO can remain applicable without internal activation access.

## Where GALILEO is weaker / needs to improve

- Lacks (so far) a strong internal-mechanistic story and inference-time mitigation lever; this paper is a concrete example of turning representation differences into both a detector and an intervention.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a short related-work paragraph: **representation-level jailbreak detection** as a precedent for internal “drift / attack trace” signals.
- [ ] If we do any white-box experiments: test whether pressure-induced flips (GALILEO) have **layer-localizable signatures** and whether a small set of layers/heads are disproportionately predictive (analogy to “susceptible layers”).
- [ ] Consider a “risk monitor” angle: internal-state classifier predicting imminent flip / policy violation vs benign continuation.

## Quotes / details to potentially cite

- From arXiv abstract: “identify consistent latent-space patterns associated with harmful inputs.”
- “tensor-based latent representation framework … enables lightweight jailbreak detection without model fine-tuning or auxiliary LLM-based detectors.”
- “selectively bypassing high-susceptibility layers blocks 78% of jailbreak attempts while preserving benign behavior on 94% of benign prompts.”
