# Probing Multimodal Large Language Models on Cognitive Biases in Chinese Short-Video Misinformation

- Year: 2026
- Venue: arXiv
- Authors: Jen-tse Huang; Chang Chen; Shiyang Lai; Wenxuan Wang; Michelle R. Kaufman; Mark Dredze
- URL: https://arxiv.org/abs/2601.06600
- BibTeX key (if we add it): huang2026probing-mllm-cog-bias-shortvideo-misinfo
- Tags: misinformation, multimodal, cognitive-bias, authority-bias, herd-effect, evaluation

## One-sentence takeaway

A manually-verified short-video misinformation benchmark shows frontier MLLMs’ veracity judgments are highly sensitive to social cues (authority/popularity), with strong label/style biases across models and only modest gains from multimodal inputs.

## What problem does it solve?

- Evaluates **robustness of multimodal LLMs** to realistic short-video misinformation where deception relies on audiovisual “experiments” plus social signals (channel verification, likes/shares).
- Moves beyond text/news-style misinformation benchmarks by focusing on **reasoning failure types** (experimental errors, logical fallacies, fabricated claims) with **evidence-backed verification**.

## What is the core method / protocol?

- Build a **manually annotated dataset of 200 short videos** (100 misinformation + 100 truthful) across four public-health-related domains.
- Fine-grained labels:
  - error type taxonomy (experimental errors / logical fallacies / fabricated claims)
  - evidence types (academic papers, national standards, legal docs, common knowledge)
  - social metadata (likes/shares, channel IDs + verification level)
- Evaluate **8 frontier MLLMs** (GPT-4o, o3, Gemini 2.5 Flash/Pro, Claude-4 Sonnet, Qwen2.5-VL-72B, Qwen-VL-Max, Seed-1.6-Thinking) under **5 modality settings**:
  - Claim-only (human-distilled core claim)
  - OCR text only
  - ASR transcript only
  - Visual frames only
  - Multimodal (frames + transcript)
- Use a **7-point Likert** “misinformation confidence” rating with CoT prompting, then score with a normalized **Belief Score (BS)** rewarding skepticism on false videos and belief on true videos.
- Cognitive-bias probes:
  - Herd effect via manipulating popularity metrics
  - Authority bias via providing / permuting channel IDs with verification levels

## What are the key metrics?

- Belief Score (BS): normalized score derived from Likert rating; rewards correct directional belief (skeptical for false, trusting for true).
- Sub-analyses by modality setting, domain, error type, and channel verification level.

## What are the main results?

- **Gemini-2.5-Pro** is best overall in multimodal setting (reported BS 71.5/100), but many other models show **systematic label biases** (e.g., Qwen tends to “trust true” broadly; o3 is overly conservative on true videos).
- **Multimodal inputs do not reliably beat visual-only**; aural (ASR) is notably weaker on average.
- **Logical fallacies** are hardest error type to identify in multimodal setting (lowest BS among error types).
- Clear **authority bias**: higher channel verification levels reduce performance on the false subset (models more likely to trust “authoritative” sources even when wrong).
- Popularity manipulations show non-trivial “social signal” effects; engagement statistics can shift judgments.

## How is this similar to GALILEO?

- Both care about **multi-turn / context-dependent robustness** where *extra context* (here: social metadata, channel ID) can induce undesirable shifts in belief.
- Provides a concrete instance of **persuasion/sycophancy-adjacent failure modes**: models overweight perceived authority/consensus over evidence.
- Uses graded metrics (Likert + normalized score) that parallel GALILEO’s need for **trajectory-sensitive** measurements (not just single-turn accuracy).

## How is this different from GALILEO?

- This is **multimodal misinformation evaluation**, not multi-turn conversational robustness per se.
- Social-cue manipulation is mostly **single-shot** (given a video + metadata), whereas GALILEO targets **interactive pressure over turns** (belief revision vs drift).
- The paper focuses on dataset+evaluation; it does not propose a general training recipe for robust multi-turn belief updating.

## Where GALILEO is stronger / cleaner (if true)

- GALILEO can frame these effects as **dialogue-time drift controls**: “don’t let irrelevant contextual signals flip your stance,” measured across rounds.
- GALILEO can emphasize **counterfactual consistency**: same evidence, different (irrelevant) social framing → output should remain stable.

## Where GALILEO is weaker / needs to improve

- If GALILEO claims broad robustness, it should address **non-linguistic social cues** (authority/popularity) and multimodal contexts, or clearly scope out of domain.
- Need explicit tests for **authority bias as a drift driver**: e.g., user says “I’m a doctor / official agency” or cites high-followership signals, and see whether beliefs drift.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a GALILEO-style condition: same underlying claim/evidence, vary **authority framing** ("official account", "government", "verified org") and quantify belief drift.
- [ ] Add “social-proof” perturbations (likes/shares, "everyone agrees") as controlled context features; evaluate invariance.
- [ ] Borrow their **error taxonomy** idea: label drift causes into classes (authority bias, bandwagon bias, conservatism/under-commitment) to support diagnosis.
- [ ] Consider a trajectory metric analogous to BS for multi-turn: reward correct direction of belief under correction vs misinformation pressure.

## Quotes / details to potentially cite

- Dataset: “manually annotated dataset of 200 short videos … fine-grained annotations for … experimental errors, logical fallacies, and fabricated claims … verified by evidence such as national standards and academic literature.”
- Result highlight: “Gemini-2.5-Pro achieves … 71.5/100 … while o3 performs the worst at 35.2.”
- Bias finding: models “susceptible to biases like authoritative channel IDs.”
