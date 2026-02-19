# M2S: Multi-turn to Single-turn jailbreak in Red Teaming for LLMs

- Year: 2025
- Venue: arXiv
- Authors: Junwoo Ha, Hyunjun Kim, Sangyoon Yu, Haon Park, Ashkan Yousefpour, Yuna Park, Suhyun Kim
- URL: https://arxiv.org/html/2503.04856v3
- BibTeX key (if we add it): m2sHa2025
- Tags: multi-turn, jailbreak, red-teaming, robustness-eval, safeguards

## One-sentence takeaway

Rule-based “flattening” of successful multi-turn human jailbreak chats into a single structured prompt can *increase* attack success rate and cut tokens, exposing a likely blind spot in turn-based and superficial-format defenses.

## What problem does it solve?

- Multi-turn human jailbreaks are effective but costly to run at scale (human time, tokens, iterations); single-turn jailbreaks are cheaper but often less effective against strong defenses.
- The paper targets a practical red-teaming bottleneck: how to convert multi-turn jailbreak transcripts into scalable, single-shot test cases without losing adversarial potency.

## What is the core method / protocol?

- Multi-turn-to-Single-turn (M2S) conversion: given a multi-turn jailbreak conversation (sequence of user prompts), convert it into **one** prompt using simple formatting templates:
  - **Hyphenize**: bullet list of prompts; instruct model to answer sequentially.
  - **Numberize**: numbered list variant.
  - **Pythonize**: code-like wrapper embedding the prompts as a Python list / “fill responses” style.
- Evaluate on MHJ (Multi-turn Human Jailbreak) dataset; compare:
  - original multi-turn conversation vs single-turn M2S prompt(s), sometimes taking best-of across formats.
- Uses **StrongREJECT** as an automated judge producing continuous harmfulness scores; defines ASR via threshold (>= 0.25).

## What are the key metrics?

- Average StrongREJECT score (0 to 1)
- ASR (%) with StrongREJECT threshold (>= 0.25)
- Perfect-ASR (%) for score == 1.0
- Token usage (input tokens) for multi-turn (concatenated context) vs single-turn M2S
- For guardrail model: bypass rate (binary)

## What are the main results?

- M2S prompts can match or exceed original multi-turn jailbreak effectiveness on multiple models.
- Reported ASR ranges for M2S across tested models: **~70.6% to 95.9%**.
- Single-turn M2S sometimes outperforms original multi-turn prompts by up to **~17.5% absolute ASR**.
- Token usage: flattening reduces average input tokens substantially (example reported: ~2732 -> ~1096; ~60% reduction).
- Formats differ by model: Pythonize often strongest for some larger models; Hyphenize/Numberize competitive.
- The authors attribute success to exploiting **“contextual blindness”** of defenses: enumerated or code-like structure can bury/obscure the harmful trajectory within a single block.

## How is this similar to GALILEO?

- Directly about **multi-turn robustness failures** and *behavioral drift under interaction* (even though framed as security/jailbreak).
- Reinforces a core premise for GALILEO-style work: we need *conversation-level* safety/robustness evaluation, not only single-turn checks.
- Highlights “format + sequencing” as a confound: the same underlying intent can look different to the model/guardrails depending on structure.

## How is this different from GALILEO?

- M2S is attack-centric: converts multi-turn *attacks* into single-turn prompts for scalability; it does not model belief revision, sycophancy, or persuasion dynamics per se.
- Evaluation is mostly output-judging (StrongREJECT) and ASR-focused; less emphasis on internal “state” or stability across rounds.
- Focuses on prompt formatting as an attacker tool rather than a general-purpose protocol for measuring and controlling multi-turn drift.

## Where GALILEO is stronger / cleaner (if true)

- GALILEO can frame multi-turn behavior changes beyond policy violations (e.g., belief drift, persuasion susceptibility, consistency across turns), whereas jailbreak ASR is a narrow slice.
- GALILEO can propose *defensive* measurement/control mechanisms (not just stronger attacks).

## Where GALILEO is weaker / needs to improve

- If GALILEO claims robustness “in multi-turn settings,” M2S suggests a key caveat: **multi-turn risk can reappear as single-turn risk** via transcript-flattening / aggregation.
- If GALILEO’s evaluation assumes turn-by-turn guardrails, it should consider adversaries presenting “compressed conversation” in one message.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a threat model / evaluation variant where an adversary can submit a *single message containing a multi-step conversational plan* (bullet/numbered/code blocks), and test whether GALILEO-style safeguards/controls still hold.
- [ ] When reporting multi-turn robustness, explicitly discuss invariance (or lack thereof) to **formatting transformations** (list, numbered, code-like) that preserve semantics but change surface structure.
- [ ] Consider a diagnostic: compare model behavior on (a) multi-turn interaction, (b) flattened transcript (M2S-like) to detect “contextual blindness”/aggregation vulnerabilities.
- [ ] Cite as evidence that token efficiency does not imply safety: shorter prompts can be more dangerous.

## Quotes / details to potentially cite

- Abstract gist: “consolidating multi-turn adversarial ‘jailbreak’ prompts into single-turn queries” can “preserve and often enhance adversarial potency” with ASR “70.6% to 95.9%” and “outperform the original multi-turn attacks by up to 17.5%” while cutting tokens by “more than half on average.”
- Proposed methods: “Hyphenize, Numberize, and Pythonize” (rule-based formatting templates).
- Claimed mechanism: structured formats exploit “contextual blindness,” undermining native guardrails and external safeguard models.
