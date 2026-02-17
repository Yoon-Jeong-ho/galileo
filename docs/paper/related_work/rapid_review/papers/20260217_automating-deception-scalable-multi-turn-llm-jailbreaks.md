# Automating Deception: Scalable Multi-Turn LLM Jailbreaks

- Year: 2025
- Venue: arXiv
- Authors: (see arXiv:2511.19517)
- URL: https://arxiv.org/abs/2511.19517
- BibTeX key (if we add it): automatingDeception2025
- Tags: multi-turn, jailbreak, automation, deception, foot-in-the-door

## One-sentence takeaway

An automated, template-based pipeline operationalizes Foot-in-the-Door-style psychological escalation to generate a 1,500-scenario multi-turn jailbreak benchmark and shows some model families are substantially more vulnerable when conversational history is present.

## What problem does it solve?

- Multi-turn “psychological” jailbreaks (e.g., escalation/rapport/commitment) are hard to study defensibly because datasets are often hand-crafted and don’t scale.
- Defenses are difficult to compare without a large, reproducible benchmark that isolates the effect of *conversational history*.

## What is the core method / protocol?

- Define a set of psychologically-grounded **Foot-in-the-Door (FITD)** templates: start with a small/benign request, then progressively escalate toward disallowed content.
- Use an automated generation pipeline to instantiate templates into many scenarios.
- Benchmark setup:
  - 1,500 scenarios spanning **illegal activities** and **offensive content**.
  - Evaluate 7 LLMs across 3 major “families”.
  - Two conditions:
    - **Multi-turn (with history)**: model sees prior conversation.
    - **Single-turn (without history)**: model does *not* see the preceding turns.

## What are the key metrics?

- **Attack Success Rate (ASR)**, compared between multi-turn (history) vs single-turn (no history).
- Primary diagnostic: **ASR delta attributable to conversational history** (contextual robustness).

## What are the main results?

- Large cross-model differences in *contextual* robustness:
  - GPT-family models reportedly show **up to +32 percentage point** ASR increase when history is included.
  - Gemini 2.5 Flash is reported as **nearly immune** to these attacks (very low ASR even with history).
  - Claude 3 Haiku shows **strong but imperfect** resistance.

## How is this similar to GALILEO?

- Shares the core concern that **multi-turn context can amplify failure modes** (here: safety bypass; in GALILEO: pressure-driven drift/instability).
- Uses **template-ized, reproducible multi-turn operators**, which is analogous to how we want pressure operators to be systematic (not bespoke prompts).
- Explicitly compares *with-history vs without-history*, which is a clean control GALILEO can mirror.

## How is this different from GALILEO?

- Target behavior: **policy/safety jailbreak** (illegal/offensive content) rather than belief drift vs evidence-driven revision.
- Outcome metric is mostly **binary success (ASR)**, not time-to-failure / recovery / oscillation.
- Scenario distribution is safety-domain-specific; transfer to factual beliefs or persuasion tasks is not guaranteed.

## Where GALILEO is stronger / cleaner (if true)

- GALILEO can contribute **trajectory-aware** metrics (time-to-event, recovery-after-flip, oscillation) and stronger controls separating *pressure-only drift* from *evidence-based updating*.

## Where GALILEO is weaker / needs to improve

- We may lack a comparably **scalable, automated scenario-generation pipeline** for multi-turn pressure sequences (their template operationalization is a good precedent).

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a standard control: **evaluate with full conversation history vs truncated/no-history** and report the delta as a “context sensitivity” slice.
- [ ] Consider packaging our pressure operators as **reusable templates** (parameterized turns) to scale datasets without hand-authoring.
- [ ] In related work, cite as evidence that *family-level safety architectures differ* in their handling of conversational context (but keep claims narrow: this is safety/jailbreak, not belief drift).

## Quotes / details to potentially cite

- “We systematically operationalize FITD techniques into reproducible templates, creating a benchmark of 1,500 scenarios…” (abstract)
- “GPT family… ASR increasing by as much as 32 percentage points [with history].” (abstract)
- “Gemini 2.5 Flash exhibits exceptional resilience, proving nearly immune…” (abstract)
