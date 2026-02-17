# Quantifying Conversation Drift in MCP via Latent Polytope

- Year: 2025
- Venue: arXiv
- Authors: Haoran Shi, Hongwei Yao, Shuo Shao, Shaopeng Jiao, Ziqi Peng, Zhan Qin, Cong Wang
- URL: https://arxiv.org/abs/2508.06418
- BibTeX key (if we add it): Shi2025SecMCP (placeholder)
- Tags: drift, multi-turn, agents, mcp, prompt-injection, tool-poisoning, security-eval

## One-sentence takeaway

SecMCP detects and *quantifies* MCP-style conversation hijacking by measuring anomalous shifts in LLM latent activation trajectories ("latent polytope"), achieving AUROC > 0.915 across several models/datasets.

## What problem does it solve?

- MCP-style agents ingest external tool outputs in a shared (non-isolated) context, enabling *indirect prompt injection* / tool poisoning.
- Existing defenses are framed as filtering/detection but (per authors) struggle with: (i) static signatures, (ii) high overhead if using LLM judges, and (iii) no notion of *degree of conversational hijacking / drift*.

## What is the core method / protocol?

- Propose **SecMCP**, which monitors the agent’s internal dynamics:
  - Represent each query/turn by **LLM activation vectors** (latent representations).
  - Model "normal" conversational dynamics as trajectories in a **latent polytope space**.
  - Flag adversarial external knowledge by **detecting deviations / anomalous shifts** in the latent trajectory (conversation drift).
- Threat taxonomy emphasizes three risks in MCP-powered agent systems:
  - hijacking
  - misleading
  - data exfiltration

## What are the key metrics?

- AUROC for detection of adversarial manipulation / drift.
- (Implicit) usability / functionality preservation ("maintaining system usability"), but the primary reported metric in the abstract is AUROC.

## What are the main results?

- Evaluated on **Llama3, Vicuna, Mistral**.
- Datasets: **MS MARCO, HotpotQA, FinQA**.
- Reported detection performance: **AUROC > 0.915** (abstract).

## How is this similar to GALILEO?

- Shared framing: **multi-turn drift as the key failure mode** (trajectory / turn-by-turn dynamics), not just single-turn accuracy.
- Emphasizes *quantification* (degree of deviation) rather than binary pass/fail.
- Relevant as a neighbor for "robustness metrics" when the conversation is influenced by adversarial context.

## How is this different from GALILEO?

- Task focus is **MCP / tool-augmented agents** and security threats (prompt injection/tool poisoning), not general multi-turn conversational robustness.
- Uses **internal activation monitoring**; GALILEO may prefer black-box/behavioral metrics depending on scope.
- Evaluation targets are security outcomes (hijack/mislead/exfiltration) rather than (e.g.) stance flip, belief revision quality, or task success stability.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO stays **black-box and model-agnostic**, it may be easier to apply across closed models and APIs (no activation access required).
- If GALILEO’s protocols separate "evidence-based updating" vs "adversarial drift", that can give clearer causal interpretation.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently lacks *agent/tool* settings, it may miss important real-world drift sources (external retrieval/tool outputs).
- If GALILEO only measures output behavior, it may have less diagnostic power than latent-trajectory methods for early detection.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a brief related-work paragraph: MCP introduces non-isolated context risks; SecMCP proposes latent-trajectory drift quantification for hijack/mislead/exfiltration detection.
- [ ] Consider a GALILEO "tool-output poisoning" setting: evaluate drift when external context is adversarially seeded (even without activation access).
- [ ] If GALILEO includes an "early-warning" concept, cite SecMCP as a representative *trajectory-monitoring* approach.

## Quotes / details to potentially cite

- "SecMCP, a secure framework that detects and quantifies conversation drift, deviations in latent space trajectories induced by adversarial external knowledge." (abstract)
- "...evaluate SecMCP on three state-of-the-art LLMs (Llama3, Vicuna, Mistral) ... demonstrating robust detection with AUROC scores exceeding 0.915..." (abstract)
- Threats named: "conversation hijacking, misinformation propagation, or data exfiltration" (abstract)
