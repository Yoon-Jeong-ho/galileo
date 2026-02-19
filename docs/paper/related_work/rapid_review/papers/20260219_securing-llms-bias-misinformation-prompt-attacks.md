# Securing Large Language Models: Addressing Bias, Misinformation, and Prompt Attacks

- Year: 2024
- Venue: arXiv (survey)
- Authors: Keyu Chen; Ming Li; Pohsun Feng; Ziqian Bi; Junyu Liu; Xinyuan Song; Qian Niu
- URL: https://arxiv.org/abs/2409.08087
- BibTeX key (if we add it): securingLLM2024chen
- Tags: safety, security, hallucination, misinformation, bias, detection, jailbreak, prompt-injection, survey

## One-sentence takeaway

A broad survey of LLM security issues—hallucination/misinformation, bias, synthetic-text detection, and prompt/jailbreak attacks—summarizing representative evaluation setups and mitigation families.

## What problem does it solve?

- Organizes a fast-growing and fragmented “LLM security” literature into a few recurring threat areas (misinformation/hallucination, bias, detection of generated text, and prompt-based attacks), to help readers navigate methods and mitigations.

## What is the core method / protocol?

- Survey / taxonomy paper (not a new benchmark).
- For hallucination: discusses detection and mitigation families such as fact-checking pipelines (e.g., atomic-fact checking like FactScore; tool-based checking like FActTool), retrieval-augmented generation, embedding-based consistency checks, and classifier/ensemble detectors.
- For bias: describes bias types and common evaluation paradigms (controlled prompts, red teaming), and mitigation levers (data pre-processing, in-training constraints/adjustments, post-processing / output filtering).
- For “LLM output detection”: reviews DetectGPT-style detection and watermarking, and caveats around cross-model generalization and adversarial conditions.
- For attacks: summarizes jailbreak/prompt injection threat models (including competition-style settings like HackAPrompt) and defense ideas.

## What are the key metrics?

- No single proposed metric suite; summarizes commonly used ones across subareas.
- For detection tasks: typical classifier metrics (accuracy / precision-recall / AUROC) and robustness under distribution shift/adversary (discussed qualitatively).
- For bias: group disparities / stereotype tests / red-team hit rates (varies by study; treated at a high level).

## What are the main results?

- Primarily qualitative synthesis rather than new empirical findings.
- Highlights that many “solutions” degrade under realistic deployment conditions (open-ended generation, domain shift, adaptive adversaries), and that detection (synthetic text) and prompt-attack defenses remain brittle.

## How is this similar to GALILEO?

- Shares the high-level motivation: LLMs can be “secure on paper” but fail in interactive use (e.g., prompt injection / jailbreaks are fundamentally interaction-driven).
- Points to evaluation practices like red teaming and multi-round attack settings, which are conceptually adjacent to GALILEO’s multi-turn pressure personas.

## How is this different from GALILEO?

- Not a protocol paper: it does not define a reproducible, ground-truth-conditioned multi-turn robustness evaluation with matched controls.
- Emphasis is broad security taxonomy (bias, detection, misinformation, attacks) rather than measuring *dynamics of correctness flips* over turns.

## Where GALILEO is stronger / cleaner (if true)

- GALILEO’s key methodological contribution—conditioning on initially-correct items, matched Neutral Re-asking Control, and explicit dynamics metrics (survival / turn-of-failure / recovery)—is orthogonal to this survey and provides a sharper, reviewer-checkable evaluation design than the survey’s aggregated descriptions.

## Where GALILEO is weaker / needs to improve

- Coverage: this survey provides a wider “security umbrella” framing (bias + detection + prompt attacks) that GALILEO may want to nod to when positioning as one slice of LLM security (multi-turn robustness against conversational pressure).

## Action items for GALILEO (experiments / method / writing)

- [ ] Related work positioning: cite this as a broad LLM security survey, then clarify that GALILEO targets a specific interactive failure mode (pressure-induced correct→incorrect flips) with ground-truth tasks and dynamics metrics.
- [ ] Consider a short paragraph in the paper distinguishing GALILEO from (i) prompt-injection/jailbreak security evaluations and (ii) hallucination detection, while noting shared concerns around multi-turn interaction.

## Quotes / details to potentially cite

- Abstract-level scope statement: the paper reviews “accuracy, bias, content detection, and vulnerability to attacks” in LLM security, including fact-checking, bias evaluation/mitigation, DetectGPT/watermarking, and jailbreak/prompt injection case studies (arXiv:2409.08087).
