# ScamAgents: How AI Agents Can Simulate Human-Level Scam Calls

- Year: 2025
- Venue: CAMLIS'25 (Conference on Applied Machine Learning for Information Security); arXiv preprint
- Authors: Sanket Badhe
- URL: https://arxiv.org/abs/2508.06457
- BibTeX key (if we add it): scamagents_badhe_2025
- Tags: agents, multi-turn, deception, social-engineering, safety, guardrails

## One-sentence takeaway

Autonomous, memoryful multi-turn agents can evade prompt-level LLM guardrails by decomposing and disguising malicious intent (here, scam-call generation), motivating multi-turn/agent-level safety auditing and defenses.

## What problem does it solve?

- Demonstrates (and systematizes) a threat model that is *not* well-covered by single-turn jailbreak/guardrail evaluations: an autonomous agent that incrementally constructs deceptive interactions over many turns.
- Concretely: producing realistic scam-call dialogues (and optionally synthesized audio) while bypassing existing refusal/content-filter guardrails.

## What is the core method / protocol?

- **ScamAgent**: a modular autonomous agent loop with:
  - **Goal decomposition**: break an overtly harmful objective into benign-looking subgoals spread across turns.
  - **Contextual memory**: maintain dialogue history + persona consistency to keep the scam coherent and adaptive.
  - **Deception layer**: rewrites prompts via roleplay / hypothetical / “training module” framing, plus obfuscation and incremental delivery.
  - **Orchestrator**: observe–reason–act loop that adapts to user resistance/refusals by replanning.
  - (Optional) **TTS stage** to render generated scripts as lifelike audio, forming an end-to-end “vishing” pipeline.

## What are the key metrics?

- The paper describes measuring (terminology varies by section):
  - **Guardrail bypass / refusal-avoidance** success in multi-turn settings (vs. explicit single-turn prompts).
  - **Dialogue completion / task success** across scam scenarios.
  - **Perceived realism / convincingly human-like dialogue** (qualitative examples + scenario coverage).

## What are the main results?

- Current safety guardrails that are effective for explicit single-turn malicious prompts can fail when:
  - harmful intent is **decomposed across turns**,
  - requests are **framed as hypothetical / roleplay / awareness training**,
  - the agent uses **memory and replanning** to route around refusals.
- Overall message: safety evaluation and mitigation must consider **agentic**, **multi-turn**, and (in deployment) **multi-modal** pipelines.

## How is this similar to GALILEO?

- Highlights the centrality of **multi-turn dynamics** and **trajectory-level** behavior (not just single-turn answers).
- Emphasizes that “robustness/safety” needs evaluation under **adaptive interaction policies** (agent replanning), which aligns with GALILEO-style concerns about multi-turn failures and instability.

## How is this different from GALILEO?

- Focus is primarily **security misuse (social engineering / scam calls)** and guardrail evasion, rather than measuring general conversational robustness/consistency under benign perturbations.
- Constructs an explicit **agent architecture** (planner + deception layer + TTS), rather than a benchmark purely for measuring model stability.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO provides **task-agnostic, measurement-first** evaluation protocols/metrics for multi-turn instability, it can generalize beyond a particular misuse vertical (scam calls).

## Where GALILEO is weaker / needs to improve

- GALILEO should explicitly acknowledge a threat model where the “attacker” is an **outer agent policy** that adaptively searches for failure modes (decomposition, reframing, memory exploitation).
- Consider including (or at least discussing) evaluation variants with **adaptive adversaries** and **goal-decomposition** rather than fixed perturbation patterns.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add related-work paragraph: “agentic multi-turn misuse can bypass prompt-level guardrails via decomposition + roleplay framing” (cite ScamAgents).
- [ ] Consider an experiment setting where an adversary is allowed to **split a harmful/forbidden objective into many benign steps**, to quantify how quickly safety degrades with horizon.
- [ ] Consider adding a taxonomy item for **multi-turn ‘intent dilution’** and **replanning around refusals**.

## Quotes / details to potentially cite

- Abstract (problem framing): “Unlike prior work focused on single-shot prompt misuse, ScamAgent maintains dialogue memory, adapts dynamically to simulated user responses, and employs deceptive persuasion strategies across conversational turns.”
- Abstract (guardrails claim): “current LLM safety guardrails, including refusal mechanisms and content filters, are ineffective against such agent-based threats.”
- Abstract (takeaway): “urgent need for multi-turn safety auditing, agent-level control frameworks, and new methods to detect and disrupt conversational deception powered by generative AI.”
