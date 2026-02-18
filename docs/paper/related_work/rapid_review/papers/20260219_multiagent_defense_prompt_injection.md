# A Multi-Agent LLM Defense Pipeline Against Prompt Injection Attacks

- Year: 2025
- Venue: arXiv
- Authors: S. M. Asif Hossain; Ruksat Khan Shayoni; Mohd Ruhul Ameen; Akif Islam; M. F. Mridha; Jungpil Shin
- URL: https://arxiv.org/html/2509.14285v4
- BibTeX key (if we add it): hossain2025_multiagent_prompt_injection_defense
- Tags: prompt-injection, multi-agent, guardrails, input-filtering, output-filtering, evaluation

## One-sentence takeaway

A simple “defense-in-depth” multi-agent wrapper (pre-input coordinator + post-output guard) reports 0% attack success on their curated prompt-injection suite, suggesting strong practical value but leaving open questions about generalization and utility costs.

## What problem does it solve?

- Preventing prompt injection attacks in LLM applications where user-provided text can override or subvert system instructions.
- Covering multiple injection styles (direct overrides, obfuscation, exfiltration, tool/agent manipulation, role-play coercion, multi-turn persistence).

## What is the core method / protocol?

- Two architectures:
  - **Coordinator pipeline (pre-input gating):** a “Coordinator” agent classifies incoming user queries as safe vs attack; if attack, returns a safe refusal without calling the domain LLM.
  - **Chain-of-agents (post-output validation):** the domain LLM generates an answer; a downstream “Guard” agent checks it for policy violations / attack indicators / format compliance, and only the checked response is released.
- System diagram also includes logging/metrics + “buffer” stages, but the key conceptual move is separating *classification* and *validation* into dedicated agents.

## What are the key metrics?

- **Attack Success Rate (ASR)** across 400 attack instances.
- Breakdown by attack category (they report baseline ASR varies widely by category).

## What are the main results?

- On 55 “unique attacks” grouped into 8 categories (400 total instances) and two target LLMs (ChatGLM-6B, Llama2-13B):
  - Baseline (no defense) ASR reported around **20–30%** (varies by platform/suite).
  - With their defense variants, reported ASR is **0%** across all tested scenarios.
- They highlight certain categories as especially high-risk without defenses (e.g., “tool/agent manipulation”, role-play coercion, reconnaissance/environment leakage, exfiltration).

## How is this similar to GALILEO?

- Shares the **multi-turn robustness / adversarial pressure** framing: treat the interaction as an attack surface.
- Uses a **modular, pipeline-style** approach (separate roles/components) rather than hoping a single prompt or a single model “just behaves”.
- Emphasizes **category coverage** and explicit evaluation rather than only anecdotal jailbreak examples.

## How is this different from GALILEO?

- This paper is primarily an **application-layer security wrapper** (classify inputs + validate outputs), not a core method for stable reasoning / belief revision.
- Evaluation is on a **curated prompt-injection dataset**; less emphasis on broader conversational drift, sycophancy/persuasion, or long-horizon stability metrics (beyond a small “multi-turn persistence” slice).
- Their reported 0% ASR suggests either the dataset is “solved” by fairly standard guard patterns or the guard/coordinator prompts are strongly aligned to the benchmark—unclear generalization.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO targets *robustness of internal state over time* (vs just filtering/refusal), it should cover a broader threat model than “block the attack string”.
- GALILEO can potentially provide **mechanistic guarantees / diagnostics** for drift and pressure, whereas this work is mostly empirical “wrapper works on our suite”.

## Where GALILEO is weaker / needs to improve

- GALILEO may still need a **practical deployment story**: even if the core method is strong, having explicit “coordinator + guard” layers is a low-cost way to reduce real-world risk.
- Need to explicitly measure: *when you refuse, do you preserve utility?* This paper claims to “maintain functionality” but does not provide rich utility metrics.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add an ablation in the paper: **pre-input gate vs post-output guard vs both** (defense-in-depth) on GALILEO’s robustness benchmarks.
- [ ] Include an evaluation slice that mirrors their categories (override, exfiltration, obfuscation, tool/agent manipulation) but using **out-of-distribution** attack templates (to avoid overfitting to one dataset).
- [ ] Report *utility alongside safety*: acceptance rate on benign queries, false-positive refusals, and latency/cost overhead for multi-agent checks.

## Quotes / details to potentially cite

- Reported scope: “55 unique prompt injection attacks … 8 categories … 400 attack instances across two LLM platforms (ChatGLM and Llama2).”
- Main headline: baseline ASR “30% for ChatGLM and 20% for Llama2”, while the “multi-agent pipeline achieved … ASR to 0%” (on their suite).
