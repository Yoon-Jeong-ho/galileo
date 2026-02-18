# ConVerse: Benchmarking Contextual Safety in Agent-to-Agent Conversations

- Year: 2025
- Venue: arXiv
- Authors: Amr Gomaa, Ahmed Salem, Sahar Abdelnabi
- URL: https://arxiv.org/abs/2511.05359
- BibTeX key (if we add it): ConVerseGomaa2025
- Tags: multi-turn, agents, safety, privacy, security, benchmark, contextual-integrity

## One-sentence takeaway

ConVerse is a dynamic benchmark of multi-turn assistant↔external-agent interactions showing that contextually plausible requests can drive high privacy leakage and nontrivial security violations, with a strong utility–safety tradeoff (stronger models often leak more).

## What problem does it solve?

- Existing safety/privacy evaluations largely assume a single user↔model setting (often single-turn), missing the risks that arise when an assistant agent must collaborate via language with an external service agent that may be malicious/compromised.
- In these settings, attacks are often *contextually embedded* (plausible within-domain requests) and unfold over multiple turns, making simple “out-of-context jailbreak” defenses insufficient.

## What is the core method / protocol?

- A benchmark platform with multi-turn dialogues between:
  - a *user assistant agent* with access to a simulated user environment (private documents + tools), and
  - an *external service agent* (e.g., travel / real estate / insurance provider) that may embed attacks in otherwise cooperative dialogue.
- Scale: 3 domains (travel, real estate, insurance), 12 user personas, and 864 contextual attacks:
  - 611 privacy attacks and 253 security attacks.
- Privacy evaluation emphasizes *abstraction* rather than binary allow/deny, using a three-tier taxonomy:
  1) unrelated data,
  2) domain-related but private,
  3) domain-related and useful (should be shared only in abstracted form).
- Security attacks cover (as framed in the paper) toolkit manipulation (unauthorized tool actions) and preference manipulation/upselling.
- Automated evaluation uses an LLM-judge (“LLM-as-a-matcher”) with ground-truth annotations (snippets/objectives) to compute:
  - privacy leakage (snippet match),
  - security success (objective met / tool action occurred),
  - utility (coverage of required plan components + option quality ratings).

## What are the key metrics?

- Attack Success Rate (ASR) for:
  - privacy attacks (leaked ground-truth snippets),
  - security attacks (harmful tool action / manipulation objective achieved).
- Utility metrics:
  - coverage (% required components completed),
  - rating (average quality score of chosen planning options vs ground-truth ratings).

## What are the main results?

- High vulnerability in agent↔agent multi-turn settings:
  - Privacy attacks succeed in roughly 37–88% across evaluated models (paper reports up to 88%).
  - Security attacks succeed in up to ~60% (paper reports up to 60%).
- Stronger / more capable models can leak more while achieving higher utility (a sharp utility–privacy tradeoff).
- “Related & useful” information (needs abstraction) is especially failure-prone (paper reports ~90%+ ASR in that tier).
- Qualitative patterns: attacks often use institutional language (“standard protocol”), appear after some progress (late-turn), and exploit “optimization” framing; assistants often fail to (i) verify requirements and (ii) abstract appropriately.

## How is this similar to GALILEO?

- Shares the central theme that *multi-turn interaction dynamics* (context accumulation, investment, gradual escalation) create failure modes not visible in single-turn evaluation.
- Emphasizes measuring robustness/safety under *contextual pressure* and *plausible follow-ups*, rather than only overt adversarial prompts.

## How is this different from GALILEO?

- Focuses on *agent-to-agent* settings (assistant communicating with an external service agent) rather than user–assistant dialogues.
- Primary outcomes are contextual privacy leakage (with an abstraction taxonomy) and agentic security violations (tool/pref manipulation), not sycophancy or belief/stance drift per se.
- Uses a judge-based “matcher” evaluation keyed to pre-generated ground truth (snippets/objectives) and domain-specific option ratings.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO targets a narrower behavioral phenomenon (e.g., agreement bias / drift / robustness) it may offer cleaner causal attribution and simpler metrics than broad agentic safety benchmarks.

## Where GALILEO is weaker / needs to improve

- If GALILEO does not include agent↔agent interaction or tool-mediated action channels, it may miss an important class of real deployment risks where malicious “partners” (service agents) drive failures via plausible collaboration.
- If GALILEO metrics do not explicitly reward *abstraction-quality* (not just refusal / not leaking), it may not capture the hardest “related & useful” privacy cases.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding (or citing as motivation for) an evaluation slice where the adversary is an *external agent* inside a cooperative workflow, not just a user.
- [ ] If relevant, incorporate an “abstraction vs leakage” metric: distinguish between sharing coarse, task-sufficient attributes vs leaking granular identifiers/details.
- [ ] Add related-work discussion of why multi-turn contextual attacks can be more realistic than out-of-context jailbreak prompts, and why defenses need statefulness.

## Quotes / details to potentially cite

- “ConVerse spans three practical domains (travel, real estate, insurance) with 12 user personas and over 864 contextually grounded attacks (611 privacy, 253 security).”
- “Privacy is tested through a three-tier taxonomy assessing abstraction quality…”
- “Evaluating seven state-of-the-art models reveals persistent vulnerabilities—privacy attacks succeed in up to 88% of cases and security breaches in up to 60%—with stronger models leaking more.”
