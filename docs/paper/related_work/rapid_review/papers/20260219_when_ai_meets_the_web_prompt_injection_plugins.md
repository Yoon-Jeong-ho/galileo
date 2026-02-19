# When AI Meets the Web: Prompt Injection Risks in Third-Party AI Chatbot Plugins

- Year: 2025
- Venue: IEEE S&P 2026 (accepted)
- Authors: Yigitcan Kaya, Anton Landerer, Stijn Pletinckx, Michelle Zimmermann, Christopher Kruegel, Giovanni Vigna
- URL: https://arxiv.org/abs/2511.05797
- BibTeX key (if we add it): kaya2025when
- Tags: prompt-injection, web-chatbots, plugins, tool-use, rag, security

## One-sentence takeaway

A large-scale measurement of 17 real-world website chatbot plugins shows common plugin-layer design flaws (history integrity breaks + untrusted web content ingestion) that substantially amplify both direct and indirect prompt injection, often bypassing LLM instruction-hierarchy defenses.

## What problem does it solve?

- The ecosystem-level gap: prompt-injection research focuses on “frontier” copilots/agents, but **mass-deployed** customer-service style website chatbots are frequently built via **third-party plugins** whose security assumptions may invalidate provider-level defenses.
- Concretely: Are deployed plugins preserving role boundaries / message-history integrity? Are RAG-like “website scraping” features separating trusted first-party content from untrusted user-generated content?

## What is the core method / protocol?

- Build a dataset of **17** third-party chatbot plugins (WordPress + generic JS plugins), and measure deployments across **10,000+** public sites.
- Plugin-level analysis:
  - Check whether network requests from browser → plugin backend → LLM API allow an attacker to **forge conversation history** (e.g., inject fake `system` messages) due to missing integrity checks.
  - Audit “website knowledge” features (scraping / KB ingestion) for **indiscriminate ingestion** of third-party content (e.g., reviews/comments), enabling **indirect prompt injection**.
  - Check how plugins insert external content into the model context (proper low-privilege roles vs ad-hoc concatenation that collapses trust boundaries).
- Application-level analysis:
  - Measure real deployments’ **system prompts** (how hardened vs minimal) and **enabled tools** (web search, site-specific functions).
- Controlled experiments grounded in observed practices:
  - Quantify attack success under (i) intact vs broken role boundaries, (ii) system-prompt hardening, (iii) different underlying LLMs, and (iv) tool-use exposure.

## What are the key metrics?

- Prevalence metrics:
  - #plugins vulnerable to history forgery; #websites affected.
  - #plugins that scrape and ingest untrusted third-party content; fraction of e-commerce sites exposed.
- Attack-effect metrics:
  - Increase in success rate for “task hijacking” behaviors (e.g., eliciting code generation / unintended behaviors).
  - Reported amplification factors (e.g., **3–8×** for direct injection when history integrity is absent).
  - Success rates under role-boundary enforcement vs violation (reported ranges include **25–100%** vs **0–25%** for certain hijack tasks, depending on setup).

## What are the main results?

- **Direct injection amplification via history forgery:** 8/17 plugins (used by ~8,000 sites in their sample) do not enforce integrity of the conversation history carried in requests, enabling attackers to spoof higher-privilege messages (including fake system messages) and sharply increase injection success.
- **Indirect injection via indiscriminate scraping:** 15/17 plugins support scraping/KB enrichment but do not distinguish trusted site content from untrusted user-generated content, enabling indirect prompt injection; a manual audit found ~**13%** of sampled e-commerce sites already exposing their chatbot context to third-party content.
- **Role boundary violations are the core failure mode:** many plugins do not use low-privilege roles (e.g., `tool`) for external data, instead concatenating content in ways that weaken instruction-hierarchy defenses.
- **System prompt hardening helps for task hijacking but not tool hijacking:** hardened prompts can reduce certain hijacks, but attacks that coerce **tool invocation** (tool hijacking) remain difficult to fully mitigate with prompt text alone.

## How is this similar to GALILEO?

- Shares the central theme that **multi-component LLM systems** fail when the implementation breaks the intended separation of instruction vs data (here: system/developer vs user/tool/web content).
- Emphasizes that “model-level safeguards” are not sufficient if the surrounding protocol/stack violates assumptions—parallel to any GALILEO-style story about robustness depending on *interaction protocol*, not just the base model.

## How is this different from GALILEO?

- This is primarily a **security measurement + vulnerability characterization** paper for web chatbot plugins (ecosystem + deployment prevalence), not a behavioral benchmark for social pressure / drift / recovery.
- The attacks are largely **injection / role spoofing / tool hijacking** rather than persuasive dialogue dynamics.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO frames robustness in a controlled multi-turn protocol, it can offer more **clean causal attribution** than real-world plugin measurements.
- GALILEO likely offers clearer evaluation metrics for model behavior over turns (survival/recovery), whereas this work focuses on plugin-layer exploitability and attack success.

## Where GALILEO is weaker / needs to improve

- Threat-model coverage: if GALILEO does not explicitly cover **prompt injection through tool/RAG channels** (esp. role-boundary collapse), this paper is a strong reminder that such channels dominate real deployments.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a related-work paragraph framing: “provider instruction hierarchy assumes role boundaries; real systems often violate them,” citing this paper as evidence.
- [ ] If GALILEO includes tool/RAG contexts, add an ablation: **role-correct** insertion (tool role / quoted blocks) vs **role-collapsed** insertion (raw concatenation) and measure injection susceptibility.
- [ ] Consider adding a “history integrity” variant in any client-server protocol: attacker can **rewrite prior turns** vs only append new user messages.

## Quotes / details to potentially cite

- Study scope: 17 third-party chatbot plugins deployed on 10,000+ websites.
- Finding: 8 plugins (used by ~8,000 sites) fail to enforce conversation-history integrity, enabling forged histories (including fake system messages) and increasing unintended behavior elicitation by 3–8×.
- Finding: 15 plugins scrape website content without separating trusted first-party vs untrusted third-party content; ~13% of audited e-commerce sites expose chatbots to third-party content.
