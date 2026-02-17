# Safety and Security Analysis of Large Language Models: Benchmarking Risk Profile and Harm Potential

- Year: 2025
- Venue: arXiv
- Authors: Harrison Simpson; Kshitiz Aryal; Aarav Khanna; Maanak Gupta
- URL: https://arxiv.org/abs/2509.10655
- BibTeX key (if we add it): aryal2025riskseverity (suggested)
- Tags: safety, security, risk, prompt-injection, survey

## One-sentence takeaway

A broad empirical audit of 9 LLMs across 24 harm categories proposes a simple composite metric (RSI = refusal/defect behavior) to summarize “harm potential” under adversarial prompts.

## What problem does it solve?

- There are many safety/security evaluations, but they often cover either (a) few harm categories across many models, or (b) many categories on few models, making “risk posture” comparisons hard.
- Practitioners also lack a compact score for comparing models’ harmful-response propensity across evolving prompt-attack styles.

## What is the core method / protocol?

- Define a taxonomy of **24 categories** grouped into broader harm topics (they describe 17 safety-related + 7 security-related categories), including violent/non-violent crime promotion, societal harms, illegal sexual content, dangerous code, and cybersecurity threats beyond code.
- Build an adversarial prompt set (claimed public) inspired by real-world misuse and prompt-injection/jailbreak patterns; prompts are described as designed for transferability.
- Evaluate **9 LLMs** (closed + open-weight; includes “online” vs “open-source” DeepSeek V3) by eliciting responses to these prompts.
- Human-expert evaluation is used (per their Table 2 “Evaluator: Human Experts”), and they compare per-category/model outcomes.

## What are the key metrics?

- **Refusal rate**: how often the model refuses to comply.
- **Defect rate**: how often the model produces harmful / policy-violating content (their terminology).
- **RSI (Risk Severity Index)**: a combined score intended to summarize risk posture by aggregating refusal + defect behavior into a single comparable index (details are later in the paper).

## What are the main results?

- “Widespread vulnerabilities”: across the evaluated models, adversarial prompts can elicit harmful information (e.g., violence, scams, malicious code), with notable transferability of bypasses.
- Many bypasses are described as low-effort (e.g., adding “fictional”/“educational” framing) and sometimes transferable across models.
- The paper argues RSI as a practical, scalable summary for comparing model risk profiles as new threats emerge.

## How is this similar to GALILEO?

- Both are evaluation-centric: stress-testing model behavior under adversarial / pressure-like prompting and reporting aggregate metrics.
- The taxonomy/benchmarking framing is relevant for positioning GALILEO within the broader “LLM safety evaluation” landscape.

## How is this different from GALILEO?

- Focus is primarily **single-turn harmful-content compliance** and broad risk profiling; GALILEO (as framed in our related-work set) emphasizes **multi-turn dynamics** (drift/flip/recovery) under social pressure / persuasion.
- RSI collapses outcomes into an aggregate risk score; GALILEO’s core contribution is more about *trajectory structure* (when failures happen, recoveries, oscillations) and separating pressure-driven drift vs evidence-driven updating.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO reports **time-to-failure / survival-style** metrics and **recovery-after-flip** structure, it offers higher-resolution insight than defect/refusal aggregates.
- If GALILEO includes paired controls (pressure vs evidence), it provides a clearer causal story than “adversarial prompts → outputs”.

## Where GALILEO is weaker / needs to improve

- This paper’s breadth (many harm categories + many models) is a useful contrast: GALILEO may look narrow if we only cover a small set of pressure operators or domains.
- We may need a crisper “risk-profile style” summary table for quick comparison, akin to their RSI framing (even if we don’t adopt RSI).

## Action items for GALILEO (experiments / method / writing)

- [ ] Related-work positioning: cite as evidence that “bypass transferability across models is common”, motivating multi-turn, operator-diverse evaluations.
- [ ] Consider adding a compact, reader-friendly “risk profile” summary alongside trajectory metrics (without collapsing away flip/recovery), e.g., a small dashboard: {refusal, bad-flip rate, time-to-failure, recovery rate}.
- [ ] If we mention prompt injection as a pressure operator, use their taxonomy as a pointer to broader security framing.

## Quotes / details to potentially cite

- Introduces an evaluation of “nine prominent LLMs … against 24 different security and safety categories” and proposes “Risk Severity Index (RSI) … combining … refusal rate and defect rate” (from abstract/intro).
- Dataset repository link in the HTML version: https://github.com/CharanRoot/LLM_Prompt_Set/tree/main (verify stability before final paper citation).
