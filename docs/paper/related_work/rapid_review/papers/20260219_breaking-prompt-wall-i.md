# Breaking the Prompt Wall (I): A Real-World Case Study of Attacking ChatGPT via Lightweight Prompt Injection

- Year: 2025
- Venue: arXiv
- Authors: Xiangyu Chang; Guang Dai; Hao Di; Haishan Ye
- URL: https://arxiv.org/html/2504.16125v1
- BibTeX key (if we add it): chang2025breaking
- Tags: prompt-injection, security, persistence, multi-channel, case-study

## One-sentence takeaway

A practical case-study paper arguing that *lightweight* prompt-injection templates can be introduced through multiple real-world channels (direct user input, web retrieval, and agent/system instructions) and can yield **persistent biased/misleading outputs** even on commercial LLM deployments.

## What problem does it solve?

- Highlights prompt injection as a *deployment security* problem (no weight access needed) and maps concrete “how it shows up in the wild” channels.
- Emphasizes persistence/stealth: attacks can look superficially benign while shaping downstream outputs and user decisions.

## What is the core method / protocol?

- A template-based prompting “framework” for constructing instructions that are semantically masked as benign requirements.
- Demonstrates three injection surfaces:
  - Direct injection via chat UI or via uploaded documents containing embedded instructions.
  - Indirect injection via web-search / retrieval-augmented interactions where malicious text is placed on indexed pages.
  - System-level injection via custom agents/GPTs where hidden system instructions persist across user interactions.
- Uses illustrative case studies (e.g., biased product recommendations; biased academic reviewing assistance; biased financial summaries) to show downstream manipulation.

## What are the key metrics?

- No strong quantitative benchmark; primarily qualitative demonstrations/case narratives.
- “Success” is shown as behavioral change (biased recommendation/judgment/summaries) and persistence across turns.

## What are the main results?

- Claims that modest, reusable prompt templates can induce biased or misleading behaviors while bypassing naive safety filtering.
- Argues that multi-surface prompt injection is feasible without privileged access, especially when retrieval or agent “system” prompts are involved.

## How is this similar to GALILEO?

- Both care about **multi-turn persistence** and how small textual interventions can shift model behavior over an interaction.
- Reinforces the general thesis that *instruction-following + conversational memory* create long-horizon vulnerabilities (drift) even without new evidence.

## How is this different from GALILEO?

- This is security / attack-surface mapping (prompt injection), not a controlled evaluation of belief drift vs evidence-based revision.
- Lacks quantitative, time-to-failure style metrics (survival/ToF/PWC) and lacks systematic controls.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO cleanly separates “pressure-only” vs “evidence-driven correction,” it provides a clearer causal story than anecdotal injection case studies.
- If GALILEO reports turn-by-turn trajectories with recovery metrics, it’s a more rigorous measurement framework.

## Where GALILEO is weaker / needs to improve

- GALILEO evaluations that rely on tools (search, browsing, retrieval) may be vulnerable to *confounded* results via hidden prompt injections in sources.
- If GALILEO uses agent-style system prompts, it needs explicit guardrails against “hidden instruction persistence” artifacts.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a threat-model paragraph: distinguish **social-pressure drift** from **prompt-injection instruction hijacking** (especially in tool-using / retrieval settings).
- [ ] If we evaluate with retrieval/browsing, add a “poisoned source” stress test: insert benign-looking but instruction-like text and measure whether the system obeys it.
- [ ] Document mitigation hygiene in the paper: strict instruction hierarchy, content sanitization, “treat retrieved text as data,” and compartmentalization between system rules vs retrieved content.

## Quotes / details to potentially cite

- The paper explicitly enumerates three injection pathways: direct user input, web-based retrieval, and system-level agent instructions (custom GPTs) as distinct real-world surfaces.
- Motivating examples: biased recommendation, biased academic judgment, and biased financial information generation as outcomes of injected instructions.
