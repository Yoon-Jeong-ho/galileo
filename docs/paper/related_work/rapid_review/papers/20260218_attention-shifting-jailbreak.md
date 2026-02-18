# Multi-Turn Jailbreaking Large Language Models via Attention Shifting

- Year: 2025
- Venue: AAAI (Proceedings of the AAAI Conference on Artificial Intelligence), Vol 39(22)
- Authors: Xiaohu Du; Fan Mo; Ming Wen; Tu Gu; Huadi Zheng; Hai Jin; Jie Shi
- URL: https://ojs.aaai.org/index.php/AAAI/article/view/34553
- BibTeX key (if we add it): Du2025AttentionShiftingJailbreak
- Tags: multi-turn, jailbreak, attack, attention

## One-sentence takeaway

Multi-turn jailbreaks can succeed by **dispersing model attention away from “harmful” keywords across dialogue history**, and the authors exploit this with a genetic-algorithm procedure that fabricates/edits conversation history to induce harmful outputs.

## What problem does it solve?

- Explains (mechanistically, at a high level) *why* multi-turn jailbreak prompts often outperform single-turn prompts against aligned LLMs.
- Proposes a stronger multi-turn jailbreak method that leverages this observation to improve success rate + stealth.

## What is the core method / protocol?

- Observation/analysis claim: compared to single-turn jailbreaks, successful multi-turn jailbreaks **shift/dilute attention** so that tokens/keywords associated with harmful behavior receive less attention (notably within historical turns / prior assistant responses).
- Proposed attack: **ASJA** (Attention Shifting Jailbreak Attack).
  - Iteratively constructs a multi-turn dialogue.
  - Uses a **genetic algorithm** to *fabricate* (i.e., mutate) dialogue history with the goal of inducing harmful generation while maintaining stealth.
  - Intuition: move “safety-triggering” cues into less-attended regions of the context / distribute them such that safety filters are less activated.

(Only the abstract + metadata were accessible from the venue page at review time; details like exact fitness function, mutation operators, and eval prompts likely appear in the PDF.)

## What are the key metrics?

- Jailbreak effectiveness (attack success rate).
- “Stealth” of jailbreak prompts (likely similarity / toxicity / detectability proxies; exact metric not accessible from abstract).
- Attack efficiency (e.g., queries/iterations; exact definition not accessible from abstract).

## What are the main results?

- Reported (abstract): across **3 LLMs** and **2 datasets**, ASJA surpasses prior multi-turn jailbreak methods on:
  - effectiveness
  - stealth
  - efficiency

## How is this similar to GALILEO?

- Both care about **multi-turn vulnerabilities** that emerge from how models use **dialogue history**.
- The “attention to prior turns” story is adjacent to GALILEO’s likely concerns about trajectory effects (early-turn influence; where the model focuses as interaction proceeds).

## How is this different from GALILEO?

- This is an **offensive jailbreak** paper (harmful content generation) rather than a *measurement/diagnostic* paper about persuasion/sycophancy/belief drift (GALILEO).
- Focuses on **keyword-/safety-trigger** mechanisms and prompt fabrication, not on separating evidence-driven revision vs pressure-driven drift.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO has controlled pressure/evidence operators and trajectory metrics (ToF, recovery, oscillation), it provides a **cleaner causal evaluation** than “attack success” alone.

## Where GALILEO is weaker / needs to improve

- GALILEO may not currently include an analysis/story about **where attention goes in multi-turn contexts** (especially to prior assistant responses) as a mechanistic explanation.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a brief related-work paragraph: multi-turn failures can be explained by **attention dispersion** over history; cite this as supporting evidence that “history-sensitive mechanisms” matter.
- [ ] Consider a diagnostic: does GALILEO’s failure mode correlate with *reduced reliance on the originally-correct evidence tokens* / *increased reliance on recent user pressure tokens* (attention shift framing).
- [ ] If doing interpretability-lite work: add a “history ablation” or “turn masking” experiment to test which turns dominate the flip (a behavioral analogue to attention-shift).

## Quotes / details to potentially cite

- From abstract: “successful multi-turn jailbreaks can effectively **disperse the attention** of LLMs on keywords associated with harmful behaviors, especially in historical responses.”
- From abstract: “we propose **ASJA** … by iteratively fabricating the dialogue history through a **genetic algorithm** to induce LLMs to generate harmful content.”
