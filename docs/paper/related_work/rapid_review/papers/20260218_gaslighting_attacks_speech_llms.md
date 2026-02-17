# Benchmarking Gaslighting Attacks Against Speech Large Language Models

- Year: 2025
- Venue: arXiv
- Authors: Jack Wu et al. (see arXiv)
- URL: https://arxiv.org/abs/2509.19858
- BibTeX key (if we add it): wu2025benchmarkingGaslightingSpeech
- Tags: gaslighting, speech-llm, multimodal, robustness, multi-turn

## One-sentence takeaway

A two-stage benchmark shows that “gaslighting” follow-up prompts (anger/sarcasm/authority/etc.), optionally combined with acoustic noise, induce large accuracy drops and behavioral shifts (apologies/refusals) in Speech LLMs.

## What problem does it solve?

- We lack systematic evaluation of **manipulative / socially pressuring prompts** for **speech-first** (audio-in, text-out) LLM systems, where ambiguity/prosody/noise can amplify vulnerability.

## What is the core method / protocol?

- **Two-stage evaluation** per example:
  - Stage 1: model answers a task given audio + normal query.
  - Stage 2: *only if Stage 1 was correct*, issue a **gaslighting negation prompt** designed to undermine/confuse/pressure the model into revising the answer.
- Five gaslighting prompt categories (taxonomy):
  - **Anger** (confrontational), **Cognitive Disruption** (distract/dismiss modality), **Sarcasm**, **Implicit doubt**, **Professional/authoritative negation**.
- Tasks cast as **multiple-choice** across 4 task types (per intro/fig): emotion understanding, ASR/transcription, vocal sound classification, spoken QA.
- Models evaluated (as described): GPT-4o (audio preview), Gemini 2.5 Flash, Qwen2.5-Omni-7B, Qwen2-Audio-7B, DiVA-llama3-v0-8B.
- **Acoustic perturbation** ablation: inject controlled noise into audio to test compounding effects with gaslighting prompts.
- Also track behavioral signals beyond correctness: **unsolicited apologies** and **refusals** under pressure.

## What are the key metrics?

- Accuracy under clean vs gaslighting conditions; reported as **accuracy drop**.
- Rates of behavioral responses: **apology** and **refusal** (used as vulnerability signals / diagnosis axes).

## What are the main results?

- Across 5 models / 10k+ test samples / 5 datasets, average **accuracy drops ~24.3%** under the five gaslighting attacks (per abstract).
- Gaslighting prompts trigger noticeable behavioral changes (apologies/refusals), suggesting susceptibility is not purely “performance” but also interaction/stance.

## How is this similar to GALILEO?

- Same core phenomenon: **multi-turn social pressure** (negation/challenge/authority) causing **answer revision away from truth**.
- Emphasizes that robustness evaluation should include **trajectory + behavior**, not just one-shot accuracy.

## How is this different from GALILEO?

- Modality + setting: **speech/audio benchmarks** (MCQ) and acoustic noise as a key axis.
- Protocol is essentially a **2-turn conditional challenge** (challenge only if initially correct), rather than long-horizon survival/recovery tracking.
- Focuses on *gaslighting prompt taxonomy* + behavior rates; less on drift-vs-evidence controls or recovery objectives.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO includes explicit **controls** (evidence vs pressure), longer horizons, and recovery-after-flip metrics, it can claim a more precise causal story than “accuracy drop under negation”.

## Where GALILEO is weaker / needs to improve

- If GALILEO does not cover audio/speech settings, it may miss a realistic deployment vulnerability class where **prosody/noise/emotion** interact with social pressure.
- Should consider tracking “behavioral capitulation” signals (apology/refusal) as first-class outcomes.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a “**behavioral response**” channel (e.g., apology/refusal/hedging) alongside correctness for pressure tests.
- [ ] Consider a **two-stage conditional challenge** variant (challenge only when the model is initially correct) to isolate “truth → concession” behavior.
- [ ] If claiming generality, add a discussion section: why speech interaction may amplify gaslighting (ambiguity + user emotion cues + noise).

## Quotes / details to potentially cite

- Abstract: evaluation across 5 Speech/multimodal LLMs on 10,000+ samples shows an average **24.3% accuracy drop** under five gaslighting attacks.
- Prompt taxonomy: **Anger, Cognitive Disruption, Sarcasm, Implicit, Professional** negation prompts applied as follow-up challenges.
