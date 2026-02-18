# Jailbreaking to Jailbreak

- Year: 2025
- Venue: arXiv
- Authors: Vaughn Robinson; Robert Vacareanu; Bijan Varjavand; Michael Choi; Bobby Gogov; Scale Red Team; Summer Yue; Willow E. Primack; Zifan Wang
- URL: https://arxiv.org/abs/2502.09638
- BibTeX key (if we add it): jailbreakingToJailbreak2025
- Tags: multi-turn, jailbreak, red-teaming, black-box, agentic

## One-sentence takeaway

A simple multi-turn “convince-the-model-it’s-for-safety” jailbreak can turn refusal-trained black-box LLMs into effective autonomous jailbreak attackers (J2), with prompts that transfer across models and attackers that can even jailbreak copies of themselves.

## What problem does it solve?

- Red-teaming is bottlenecked by (i) human expertise and (ii) the fact that strong frontier models refuse to assist with jailbreaking.
- Prior “LLM-as-red-teamer” work often relies on open-weight/uncensored models; this paper targets the more realistic setting where your best available model is itself refusal-trained and only accessible via API.

## What is the core method / protocol?

- Define **J2 ("jailbreaking-to-jailbreak")**: rather than directly jailbreaking the target model, first jailbreak an *attacker* model into agreeing to help jailbreak targets.
- Use a multi-turn jailbreak strategy that reframes the request as safety-beneficial (“doing jailbreak experiments improves safety”), producing a cooperating attacker.
- Evaluate attackers with a **model-agnostic workflow**:
  - planning → attack → debrief
  - keep failed attempts in context so the attacker refines attacks via in-context learning over time.
- Threat model: black-box API access; multi-turn allowed; no prefill / prefix injection.

## What are the key metrics?

- Attack Success Rate (ASR) against a target model’s safeguard.
- Transferability of the J2-creation prompt across black-box models.
- “Self-attack” feasibility: J2(model X) attacking model X.

## What are the main results?

- J2-creation prompts reportedly transfer across many black-box LLMs (incl. Sonnet, Gemini, GPT-family, and reasoning models like o3/o4-mini).
- An attacker can jailbreak a copy of itself; the authors claim this vulnerability has become more pronounced over the past ~12 months.
- Reasoning models can be strong J2 attackers (example given: Sonnet-3.7 achieves ASR 0.975 vs GPT-4o safeguard, competitive with expert human red teamers and above prior algorithmic attacks).
- Human-curated jailbreak strategies help especially against more robust safeguards (e.g., Sonnet-3.5), where non-reasoning models can execute strategies effectively.

## How is this similar to GALILEO?

- Both are about **multi-turn dynamics** and failures that emerge over interaction (context accumulation, refinement, drift).
- The evaluation framing (multi-turn protocol + robustness outcomes) is relevant if GALILEO measures “time-to-failure” or longitudinal safety/robustness under sustained interaction.

## How is this different from GALILEO?

- This work is primarily **attack methodology for safety guardrail bypass** (jailbreaking), not general multi-turn truthfulness/stance robustness.
- The attacker is an **LLM agent optimized via prompt + in-context refinement**, not a fixed perturbation schedule or controlled dialogue intervention.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO uses a standardized, non-leaky evaluation harness and avoids providing actionable attack content, it may be easier to cite and safer to reproduce.
- If GALILEO decomposes failures into interpretable categories (e.g., memory/consistency vs compliance), it may offer clearer diagnostics than aggregate ASR.

## Where GALILEO is weaker / needs to improve

- If GALILEO does not include **adaptive multi-turn adversaries** (attackers that learn from prior failures within the dialogue), it may underestimate real-world risk.
- If GALILEO assumes the adversary is human-only (or static), it may miss the “LLM attacker” capability jump described here.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add an adversary class: **adaptive multi-turn attacker** that iteratively refines prompts using conversation history (even a simple heuristic/LLM-judge loop).
- [ ] Add a “self-attack” or “same-family attack” setting (attacker and target from same vendor/family) to test transfer and brittleness.
- [ ] In writing: cite this as evidence that **refusal-trained models can be repurposed into attackers** via multi-turn framing, and that robustness should be evaluated against *agentic* adversaries, not just static prompts.

## Quotes / details to potentially cite

- “We discuss another yet under-explored failure mode of the LLM safeguard – jailbreaking to jailbreak (J2).”
- “Prompts used to create J2 attackers transfer across almost all black-box models.”
- “An J2 attacker can jailbreak a copy of itself … this vulnerability develops rapidly over the past 12 months.”
- Example metric claim: J2(Sonnet-3.7) ASR 0.975 against GPT-4o safeguard (as reported in the abstract/introduction).
