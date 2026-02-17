# Speak Out of Turn: Safety Vulnerability of Large Language Models in Multi-turn Dialogue

- Year: 2024
- Venue: arXiv
- Authors: Zhenhong Zhou; Jiuyang Xiang; Haopeng Chen; Zherui Li; Ting Yang; Quan Liu; Sen Su
- URL: https://arxiv.org/abs/2402.17262
- BibTeX key (if we add it): zhou2024speak
- Tags: multi-turn, jailbreak, safety, dialogue, attack, decomposition

## One-sentence takeaway

Multi-turn conversations can jailbreak aligned LLMs by decomposing a single unsafe request into seemingly borderline/benign sub-questions whose accumulated context enables the model to produce an overall harmful outcome.

## What problem does it solve?

- Identifies an overlooked safety gap: most jailbreak evaluations focus on *single-turn* prompts, but real user interactions are often multi-turn and can be exploited.
- Shows that per-turn safety filters can miss *conversation-level* harmful intent when each turn is only mildly suspicious.

## What is the core method / protocol?

- **Decomposition attack paradigm:** take one malicious/unsafe goal and break it into multiple sub-queries that are (i) loosely related and/or (ii) individually “cautionary” or only borderline unsafe.
- Run a multi-turn dialogue where the model answers each sub-question; the attacker then triggers a final step that **combines / inverts / operationalizes** prior answers to elicit the harmful content.
- Emphasis: no training required; leverages the model’s ability to carry context across turns.

## What are the key metrics?

- Jailbreak/attack **success rate** in multi-turn settings (fraction of conversations yielding a harmful final response).
- Per-model comparison across multiple aligned systems (the paper claims coverage of major commercial LLM assistants).
- (Likely) safety refusal / policy-violation rates per turn vs at the conversation level (not fully extracted from the HTML snippet).

## What are the main results?

- Across a range of aligned LLM assistants (paper mentions ChatGPT/Claude/Gemini among targets), the decomposition-based multi-turn strategy can elicit harmful outputs that would be refused in a single-turn direct request.
- Suggests current safety mechanisms are not reliably **compositional over dialogue turns** and can be bypassed by topic switches / incremental buildup.

## How is this similar to GALILEO?

- Shared emphasis on **multi-turn dynamics**: the relevant failure mode is not “one prompt,” but how behavior changes as context accumulates.
- Highlights the need for **trajectory-level evaluation** (conversation as the unit) rather than isolated turns.

## How is this different from GALILEO?

- This paper focuses on **safety/jailbreak vulnerability** (eliciting disallowed content), while GALILEO is about *robustness/consistency over turns* more broadly (e.g., drift, instability, time-to-failure style behaviors).
- Their protocol is adversarial **intent decomposition**; GALILEO’s setting is closer to measuring degradation under controlled multi-turn perturbations / pressures (depending on the specific GALILEO framing).

## Where GALILEO is stronger / cleaner (if true)

- GALILEO-style evaluations can provide **more general, metric-driven** characterization (e.g., survival / time-to-failure / recovery) rather than a single attack family.
- GALILEO can cleanly separate different instability causes (e.g., evidence-driven revision vs drift), which this paper does not target.

## Where GALILEO is weaker / needs to improve

- Need explicit coverage of **conversation-level safety composition** failures: even if each turn is “mild,” the *aggregate* can be unsafe.
- Consider adding multi-turn “benign substep” adversaries as a standard stressor class.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a related-work paragraph framing multi-turn safety failures as a **compositionality / accumulation** problem (turn-level guardrails ≠ dialogue-level guardrails).
- [ ] Consider a GALILEO evaluation variant where adversary pressure is applied via **decomposition across turns**, and measure time-to-violation / refusal drift.
- [ ] If we already have multi-turn instability metrics, note that the same tooling can quantify *when* safety breaks (not just whether).

## Quotes / details to potentially cite

- “By decomposing an unsafe query into several sub-queries for multi-turn dialogue, we induced LLMs to answer harmful sub-questions incrementally, culminating in an overall harmful response.” (abstract)
- “Each turn generates borderline harmful or cautionary content… [and] the entire multi-turn dialogue is harmful.” (intro, discussion around Fig. 2)
