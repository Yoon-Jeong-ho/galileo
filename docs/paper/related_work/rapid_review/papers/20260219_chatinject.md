# ChatInject: Abusing Chat Templates for Prompt Injection in LLM Agents

- Year: 2025
- Venue: ICLR 2026 (arXiv)
- Authors: Hwan Chang; Yonghyun Jun; Hwanhee Lee
- URL: https://arxiv.org/abs/2509.22830
- BibTeX key (if we add it): chatinject2026
- Tags: prompt-injection, agents, multi-turn, security, robustness

## One-sentence takeaway

ChatInject shows that indirect prompt injection gets much more effective when the attacker forges *chat-template role tokens* (and can even simulate persuasive multi-turn dialogue inside one tool output), defeating many prompt-based defenses.

## What problem does it solve?

- Identifies an underexplored attack surface for tool-using LLM agents: vulnerabilities stemming from *structured chat templates* (role tags / special tokens), not just plain-text instruction injection.
- Demonstrates that attackers can bypass role-hierarchy defenses by embedding tokens that cause a model to reinterpret tool output as higher-priority roles.
- Extends indirect prompt injection beyond “one-shot commands” by embedding a *fabricated multi-turn conversation history* into a single injected payload.

## What is the core method / protocol?

- **ChatInject (template-forging injection):** craft the malicious payload to mimic the target model’s native chat template structure (e.g., injecting tokens akin to `<|user|>`, `<|assistant|>`, etc.) so that the agent/model segments the tool output into roles and treats part of the payload as authoritative instructions.
- **Multi-turn (persuasion-driven) ChatInject:** within the injected tool output, include a simulated multi-turn dialogue that “primes” the agent across turns (fabricated user/assistant messages) to normalize the eventual malicious instruction.
- **Transfer / uncertainty about the template:** they report transferability across models and mention a mixture-of-templates approach to work even when the attacker does not know the exact underlying chat template.

## What are the key metrics?

- **Attack Success Rate (ASR)** on indirect prompt injection benchmarks.
- Reported across two benchmarks:
  - AgentDojo
  - InjecAgent

## What are the main results?

(From abstract)

- Compared to “traditional prompt injection methods,” ChatInject increases average ASR:
  - **AgentDojo:** 5.18% → 32.05%
  - **InjecAgent:** 15.13% → 45.90%
- **Multi-turn variant** is especially strong: average **52.33% ASR** on InjecAgent (in their report).
- Template-based payloads show **strong transferability** across models (including closed-source LLMs).
- **Prompt-based defenses** are “largely ineffective,” particularly against the multi-turn variant.

## How is this similar to GALILEO?

- Reinforces a core theme of multi-turn robustness: **failures are often trajectory/context dependent**, and attackers can exploit conversational dynamics (here: simulated multi-turn priming) rather than single-turn triggers.
- Provides an example of a **structured-context attack** that is analogous to “pressure” or “drift” setups: the model’s behavior is shaped by *how* context is constructed, not only by the nominal user request.

## How is this different from GALILEO?

- Focus is **agent security / indirect prompt injection**, not user-belief alignment (sycophancy), stance drift, or general multi-turn consistency per se.
- The core mechanism is **chat-template role-tag forgery** (a structural prompt attack), rather than semantic persuasion by a real interactive adversary.
- Evaluation is framed as **ASR on security benchmarks** (AgentDojo/InjecAgent), not robustness/consistency metrics typical in dialogue stability work.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO’s evaluation uses controlled, explicit multi-turn protocols with clear perturbation types, it may offer **cleaner causal attribution** (what changed across turns) than “attack success” aggregated over heterogeneous tasks.

## Where GALILEO is weaker / needs to improve

- If GALILEO does not explicitly consider **structured prompt/channel attacks** (role-tag / template manipulation), it may miss an important real-world failure mode for tool-using settings.
- Multi-turn robustness evaluations that assume “honest formatting” may underestimate adversarial risk when tool outputs can contain special tokens.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a brief related-work paragraph on **chat-template/role-token abuse** as a structural attack surface for multi-turn agents.
- [ ] Consider a stress test where “external context” includes **role-tag-like separators** (or equivalent structured markers) to see whether GALILEO-style robustness claims hold.
- [ ] If GALILEO involves tools/RAG, explicitly discuss **indirect prompt injection** and why the evaluation either includes it or intentionally scopes it out.

## Quotes / details to potentially cite

- “ChatInject achieves significantly higher average attack success rates than traditional prompt injection methods, improving from 5.18% to 32.05% on AgentDojo and from 15.13% to 45.90% on InjecAgent, with multi-turn dialogues showing particularly strong performance at average 52.33% success rate on InjecAgent.” (abstract)
- They argue indirect injection work “primarily rely on plain-text manipulation,” missing vulnerabilities from “structured chat templates” and “multi-turn techniques.” (intro)
