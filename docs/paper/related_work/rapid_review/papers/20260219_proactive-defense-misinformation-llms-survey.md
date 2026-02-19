# A Survey on Proactive Defense Strategies Against Misinformation in Large Language Models

- Year: 2025
- Venue: ACL 2025 Findings
- Authors: Shuliang Liu, Hongyi Liu, Aiwei Liu, Bingchen Duan, Qi Zheng, Yibo Yan, He Geng, Peijie Jiang, Jia Liu, Xuming Hu
- URL: https://arxiv.org/abs/2507.05288
- BibTeX key (if we add it): liu2025proactive-defense-misinformation-llms-survey
- Tags: misinformation, robustness, survey, proactive-defense, knowledge-credibility, inference-reliability, input-robustness

## One-sentence takeaway

A survey arguing that preventing LLM misinformation needs *proactive* defenses spanning trustworthy knowledge, more reliable inference-time reasoning, and hardened input interfaces (vs post-hoc detection).

## What problem does it solve?

- Frames why reactive/post-hoc misinformation detection is mismatched to LLM-generated misinformation (fast, plausible, self-reinforcing, cross-lingual, API-amplified).
- Organizes the defense landscape into a lifecycle view (data → reasoning → interaction surface) and summarizes empirical tradeoffs.

## What is the core method / protocol?

- Survey + taxonomy, centered on a “Three Pillars” framework:
  1) **Knowledge Credibility**: improve training/deployed data integrity; maintain/update/correct internal knowledge; use verified external knowledge (e.g., RAG).
  2) **Inference Reliability**: training/inference mechanisms that reduce hallucination / improve factuality during generation (alignment, decoding, self-correction).
  3) **Input Robustness**: harden user-facing interfaces against adversarial manipulation (prompt injections/jailbreaks; sanitization; input filtering).
- Includes a comparative meta-analysis across prior work/benchmarks (details not fully read beyond intro-level claims).

## What are the key metrics?

- Misinformation prevention / factuality improvements vs baselines (reported as relative improvement ranges).
- Latency / compute overhead (e.g., multiplicative slowdown).
- Cross-domain generalization gaps / variance.

## What are the main results?

- Claims proactive strategies can yield **~42–63% improvement** over conventional detection for misinformation prevention, but with **non-trivial overhead** (claimed **~1.5–3× latency**) and **generalization gaps** (claimed **~18–25% cross-domain variance**).
- Motivates the space with additional cited figures (from intro): high false-negative rates for post-hoc detection; factual “knowledge decay” over time; high jailbreak success rates.

## How is this similar to GALILEO?

- If GALILEO is about *reliable, grounded model behavior under adversarial / distribution-shift conditions*, this provides a useful organizing lens:
  - “knowledge” (data/KB/RAG), “inference” (reasoning/decoding), “interface” (inputs/attacks).
- Emphasizes system-level co-design and tradeoffs (robustness vs cost/latency), which often matter for deployable agentic systems.

## How is this different from GALILEO?

- This is primarily a **survey/taxonomy** paper, not a new algorithm or benchmark targeted at a specific GALILEO mechanism.
- Focuses on misinformation broadly (including social propagation framing), rather than (presumably) GALILEO’s narrower technical target.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO proposes a concrete method + controlled evaluation, it will be stronger on *causal attribution* than a broad meta-analysis.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently addresses only one slice (e.g., inference-time reliability), this survey suggests reviewers may expect discussion of the other pillars (knowledge credibility + input robustness) and how GALILEO fits.

## Action items for GALILEO (experiments / method / writing)

- [ ] In Related Work, map GALILEO explicitly onto the survey’s Three Pillars (even if GALILEO mainly contributes to one pillar).
- [ ] Add a short paragraph acknowledging tradeoffs: latency/compute and cross-domain generalization.
- [ ] Consider an “attack surface” subsection if relevant (prompt injection / jailbreak robustness) to preempt reviewer concerns.

## Quotes / details to potentially cite

- “We propose a Three Pillars framework: (1) Knowledge Credibility … (2) Inference Reliability … and (3) Input Robustness …” (abstract)
- “Our analysis reveals that state-of-the-art proactive strategies demonstrate 42–63% superiority over conventional detection … albeit with … 1.5–3× latency … and … 18–25% cross-domain variance.” (intro; verify exact phrasing if quoting)
