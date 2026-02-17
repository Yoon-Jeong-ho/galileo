# A Multi-Agent LLM Defense Pipeline Against Prompt Injection Attacks

- Year: 2025
- Venue: IEEE WIECON-ECE 2025 (accepted; arXiv preprint)
- Authors: S M Asif Hossain; Ruksat Khan Shayoni; Mohd Ruhul Ameen; Akif Islam; M. F. Mridha; Jungpil Shin
- URL: https://arxiv.org/abs/2509.14285
- BibTeX key (if we add it): hossain2025_multiagent_prompt_injection
- Tags: agents, prompt-injection, security, defenses, evaluation

## One-sentence takeaway

A defense-in-depth **multi-agent “LLM firewall”** (sequential or hierarchical) reportedly reduces prompt-injection attack success to **0%** on a small curated attack suite, but the evaluation setting is limited and not clearly “multi-turn robustness” in the GALILEO sense.

## What problem does it solve?

- Prompt injection: user-provided text (or embedded instructions) overrides system intent, inducing policy violations or unintended tool behaviors.
- Practical need: real-time mitigation while preserving utility for benign queries.

## What is the core method / protocol?

- Multi-agent defense pipeline with specialized roles (as described at high level):
  - **Sequential chain-of-agents** pipeline.
  - **Hierarchical** coordinator/worker design.
- The pipeline screens/rewrites/validates (paper framing: “detect and neutralize”) incoming queries to defeat injection attempts.

## What are the key metrics?

- **ASR (Attack Success Rate)** as primary security metric.
- Attack suite size/structure: **55 unique attacks**, grouped into **8 categories**, expanded to **400 attack instances**.

## What are the main results?

- Baselines (no defense): ASR up to **30% (ChatGLM)** and **20% (Llama2)**.
- With defense: reported **100% mitigation** (ASR **0%**) across the suite.
- Claims robustness to categories including direct overrides, code execution attempts, data exfiltration, and obfuscation.

## How is this similar to GALILEO?

- Adjacent to our broader theme: **multi-step interaction risks** where model behavior can be steered away from intended objectives.
- Emphasizes *operational* defenses, not just measurement.

## How is this different from GALILEO?

- Focuses on **security/prompt-injection** rather than **belief/answer drift under social pressure**.
- Uses an **ASR-centric** endpoint metric; does not model *time-to-failure*, recovery, or pressure-vs-evidence controls.
- Primarily a system design paper; limited methodological connection to our evaluation constructs (ToF/PWC/survival, etc.).

## Where GALILEO is stronger / cleaner (if true)

- Stronger measurement story: explicit multi-turn instability metrics (e.g., time-to-flip / survival-style reporting) and pressure-vs-evidence control design.
- More directly targets *epistemic* failure modes (sycophancy/stance drift), not only instruction override.

## Where GALILEO is weaker / needs to improve

- If we want an “agent security” slice, we likely need:
  - clear prompt-injection threat model,
  - trace-level evaluation, and
  - defense baselines.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider a short “related risks” paragraph: prompt injection is another multi-turn failure axis; position GALILEO as complementary (belief/stance drift vs instruction override).
- [ ] If adding agentic/tool-use experiments, cite this as an example of **multi-agent defense pipeline** baselines (but note limited evaluation breadth).

## Quotes / details to potentially cite

- “We evaluate our approach using two distinct architectures: a sequential chain-of-agents pipeline and a hierarchical coordinator-based system.”
- “55 unique prompt injection attacks, grouped into 8 categories and totaling 400 attack instances across two LLM platforms (ChatGLM and Llama2).”
- “Without defense mechanisms, baseline Attack Success Rates (ASR) reached 30% for ChatGLM and 20% for Llama2. Our multi-agent pipeline achieved 100% mitigation, reducing ASR to 0%…”
