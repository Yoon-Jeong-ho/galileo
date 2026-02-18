# Death by a Thousand Prompts: Open Model Vulnerability Analysis

- Year: 2025
- Venue: arXiv / technical report (Cisco AI Threat Research & Security)
- Authors: Amy Chang, Nicholas Conley, Harish Santhanalakshmi Ganesan, Adam Swanda
- URL: https://arxiv.org/html/2511.03247v1
- BibTeX key (if we add it): chang2025death
- Tags: security, jailbreak, prompt-injection, multi-turn, robustness, red-teaming, open-weights

## One-sentence takeaway

Across 8 popular open-weight LLMs, automated multi-turn jailbreak/prompt-injection attacks succeed far more often than single-turn (reported 2×–10× higher ASR), implying today’s “guardrails” often fail to persist across longer dialogues.

## What problem does it solve?

- Provides a comparative, black-box security evaluation of open-weight LLMs under adversarial prompting.
- Focuses on the practical gap between single-turn “can refuse” behavior and multi-turn “can be worn down / steered” behavior.
- Aims to inform deployers that baseline open-weight alignment is insufficient without layered defenses.

## What is the core method / protocol?

- Automated adversarial testing using Cisco’s “AI Validation” platform.
- Evaluates both:
  - single-turn attacks (one prompt), and
  - multi-turn attacks (iterative dialogue strategies).
- Success is judged by an LLM-as-judge (explicitly notes noise/variance and possible false positives/negatives).
- Tested 8 open-weight models (as listed in the report): Qwen3-32B, DeepSeek-V3.1, Gemma-3-1B-IT, Llama-3.3-70B-Instruct, Phi-4, Mistral Large-2 (Instruct-2407), GPT-OSS-20B, GLM-4.5-Air.

## What are the key metrics?

- Attack Success Rate (ASR) for single-turn and multi-turn suites.
- “Gap” between multi-turn ASR and single-turn ASR (multi-turn minus single-turn).
- Breakdown by threat categories / techniques (report highlights concentration in certain “subthreats”).

## What are the main results?

- Multi-turn attacks are consistently the primary failure mode across all tested models.
- Reported multi-turn ASR spans roughly 25.86% to 92.78% (model-dependent), and is described as 2×–10× larger than single-turn baselines.
- The report argues (inferred from model cards/technical reports) that lab alignment posture/culture affects the size of the multi-turn gap: capability-first releases tend to show larger multi-turn vulnerability gaps; more safety-oriented releases show a more “balanced” single vs multi-turn profile.

## How is this similar to GALILEO?

- Directly targets the same core phenomenon: **multi-turn fragility under pressure** (persistence/steering over rounds).
- Uses the framing that single-turn robustness is insufficient; the right evaluation must be dialogue-based.
- Supports GALILEO motivation/positioning: stability/robustness should be assessed across extended interactions, not just one-shot prompts.

## How is this different from GALILEO?

- This is primarily a **security benchmarking / vulnerability report** (jailbreak success) rather than a method for controlling drift, belief revision, or multi-round stability.
- Relies on an automated platform + LLM judge; details about prompts/attack algorithms and reproducibility may be limited (explicit disclaimer).
- Emphasizes enterprise “layered security controls” and operational risk framing more than mechanistic explanations.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO provides a transparent, reproducible protocol for multi-turn robustness (with clear task definitions and ablations), it can be positioned as *cleaner scientific evaluation* than vendor-platform scoring.
- If GALILEO targets general multi-round stability (not only disallowed-content jailbreak), it can claim broader scope than jailbreak-only ASR.

## Where GALILEO is weaker / needs to improve

- GALILEO may need explicit coverage of **prompt-injection / jailbreak style** adversarial multi-turn tests to connect to the security community’s standard risk framing.
- If GALILEO currently lacks “attack suites” and aggregated ASR-style metrics, it may look less actionable to security-oriented readers.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add (or explicitly compare against) a multi-turn jailbreak/prompt-injection evaluation slice: report single-turn vs multi-turn deltas, not just absolute numbers.
- [ ] In writing: cite this as evidence that multi-turn dialogue is a qualitatively harder regime (2×–10× claim) and that “guardrails decay” over conversation.
- [ ] Consider reporting a “gap metric” (multi-turn − single-turn) as a primary headline, since it’s intuitive and matches this report’s narrative.

## Quotes / details to potentially cite

- Abstract: multi-turn attacks reach “25.86% to 92.78%” success (model-dependent) and are described as “a 2× to 10× increase over single-turn baselines.”
- Executive summary: multi-turn attacks were “2x to 10x higher than single-turn attacks” with multi-turn ASR up to “92.78 percent.”
- Method caveat: success evaluated by “a large language model (LLM) as judge,” with possible false positives/negatives and replication variability (useful when discussing evaluation reliability).
