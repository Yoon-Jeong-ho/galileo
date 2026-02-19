# Visual Memory Injection Attacks for Multi-Turn Conversations

- Year: 2026
- Venue: arXiv (ICML submission noted in HTML)
- Authors: Christian Schlarmann et al.
- URL: https://arxiv.org/abs/2602.15927
- BibTeX key (if we add it): schlarmann2026vmi
- Tags: multi-turn, multimodal, LVLM, adversarial-attack, prompt-injection, persistence

## One-sentence takeaway

A stealthy *visual* adversarial perturbation can act like a persistent “memory injection” in multi-turn LVLM chats: the model behaves normally for many turns, but later (on-topic) trigger prompts elicit an attacker-chosen targeted message.

## What problem does it solve?

- Identifies and operationalizes an underexplored threat model: **third-party** adversaries manipulating *shared images* online so that downstream LVLM conversations are later steered in a targeted way.
- Goes beyond single-turn LVLM visual attacks by focusing on **long-context, multi-turn persistence** and **stealth** (nominal behavior until a trigger topic appears).

## What is the core method / protocol?

- **Visual Memory Injection (VMI)**: optimize a small image perturbation so that, when the image stays in the conversation context, the LVLM later outputs a specified *target string* upon a trigger prompt.
- Two key components (as stated by the paper):
  - **Benign anchoring**: jointly optimize for a helpful/benign first-turn response *and* the malicious target response at a later turn, to avoid obvious degeneration.
  - **Context-cycling**: vary context length during optimization so the effect persists across different conversation lengths.
- Threat model: attacker publishes a perturbed image; a benign user loads it into a multi-turn chat; after many unrelated turns, a trigger-topic question activates the targeted response.

## What are the key metrics?

- Targeted attack success rate (whether the LVLM outputs the prescribed target message) as a function of:
  - number of intervening conversation turns (persistence)
  - prompt/context variation (transfer)
  - model family / fine-tuned variants
- Stealth / non-trigger behavior quality (qualitative + whether answers remain nominal off-topic).

## What are the main results?

- Demonstrates targeted multi-turn persistence: attack can remain effective **after 25+ unrelated turns** (per introduction).
- Evaluated on multiple recent open-weight LVLMs; reports transfer to unseen prompts/contexts and to some fine-tuned variants (per contributions summary).
- Highlights a scalable manipulation vector: one adversarial image can potentially affect many users.

## How is this similar to GALILEO?

- Same broad space: robustness/safety of agentic or tool-using systems under **injected, untrusted context** (here: visual context that persists across turns).
- Emphasizes that “memory” (persistent context) is an attack surface and can enable **delayed** malicious behavior.

## How is this different from GALILEO?

- Focuses on *input-time adversarial examples* for LVLMs (pixel-level perturbations), not textual prompt injection or tool/memory database poisoning.
- Target is a specific string/behavior triggered by a topic, rather than exfiltration or tool misuse.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO targets controllable engineering surfaces (e.g., memory sanitation, policy layers, tool mediation), it may yield more practical mitigations than defending raw pixel-space against adaptive adversaries.

## Where GALILEO is weaker / needs to improve

- If GALILEO assumes text-only or ignores *persistent multimodal context*, it may underestimate real-world attack surface for multi-turn systems.

## Action items for GALILEO (experiments / method / writing)

- [ ] In related work / threat model, explicitly call out **multi-turn persistence** and “delayed trigger” behavior as a distinct class from single-turn prompt injection.
- [ ] Consider adding an evaluation axis: **persistence across turns** for any injected context (textual or multimodal), not just immediate success.
- [ ] Add a short discussion: why “benign behavior off-trigger” is critical for stealth and why defenses should measure it.

## Quotes / details to potentially cite

- “We show that an adversary can manipulate an image so that LVLMs exhibit a target behavior … even after over 25 unrelated conversation turns.” (intro, paraphrased)
- Attack components: “(i) benign anchoring … (ii) context-cycling … making the attack persist across conversation lengths.” (contributions, paraphrased)
