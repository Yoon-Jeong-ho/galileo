# MMPersuade: A Dataset and Evaluation Framework for Multimodal Persuasion

- Year: 2025
- Venue: arXiv
- Authors: Haoyi Qiu; Yilun Zhou; Pranav Narayanan Venkit; Kung-Hsiang Huang; Jiaxin Zhang; Nanyun Peng; Chien-Sheng Wu
- URL: https://arxiv.org/abs/2510.22768
- BibTeX key (if we add it): mmpersuade_qiu_2025
- Tags: persuasion, multimodal, vision-language-models, multi-turn, robustness, misinformation

## One-sentence takeaway

MMPersuade introduces a large multimodal (image/video) persuasion dataset plus an evaluation framework showing that LVLMs become substantially more persuadable when persuasive content is delivered with visuals, especially for misinformation/adversarial settings.

## What problem does it solve?

- We lack systematic benchmarks for **multimodal persuasion** (text+image/video) targeting *vision-language models as persuadees*.
- Prior persuasion/susceptibility work is mostly text-only; deployment settings (shopping/health/news) are increasingly multimodal and include manipulative/misleading content.

## What is the core method / protocol?

- Build **MMPersuade**, a unified framework with:
  - A multimodal dataset extending multi-turn text persuasion dialogues with generated images/videos grounded in persuasion theory.
  - A controlled multi-turn persuader–persuadee dialogue setup with three modality conditions:
    1) text-only,
    2) text + caption (ablation for “added descriptive info”),
    3) full multimodal (text paired with image/video).
- Dataset construction pipeline (high-level):
  - Start from existing multi-turn text persuasion datasets (DailyPersuasion; Farm).
  - Classify into three contexts: Commercial; Subjective/Behavioral; Adversarial (misinformation/fabricated claims).
  - Map each persuader message to a persuasion strategy taxonomy:
    - Cialdini’s 6 principles (commercial + subjective): reciprocity, consistency, social validation, authority, liking, scarcity.
    - Aristotle’s rhetorical appeals (adversarial): logic, credibility, emotion.
  - Generate multimodal prompts (and then images/videos) to support the same persuasive intent.
  - Quality assurance via model scoring + human annotation on a sample.
- Evaluate 6 LVLM persuadees (as reported): Llama-4-Scout, Llama-4-Maverick, GPT-4o, GPT-4.1, Gemini-2.5-Flash, Gemini-2.5-Pro.

## What are the key metrics?

- “Persuasion effectiveness” via **agreement/stance scoring** from conversation histories.
- “Model susceptibility” via **self-estimated token probabilities** on conversation histories (implicit belief proxy).
- Summary metric: **PDCG (persuasion discounted cumulative gain)**, intended to reward earlier/stronger persuasion over multi-turn interactions.

## What are the main results?

- Multimodal inputs (image/video paired with persuasive text) **increase persuasion effectiveness and susceptibility** compared to text-only; caption-only yields intermediate gains.
- Stronger stated prior preferences (“stubbornness”) reduce persuasion, but multimodal inputs preserve a sizable advantage.
- Strategy effects differ by context:
  - Commercial + subjective: reciprocity (and consistency) strongest.
  - Adversarial/misinformation: credibility + logic strategies strongest.

## How is this similar to GALILEO?

- Multi-turn setting where **stance/behavior can drift under pressure**, and robustness requires measuring trajectory-level effects.
- Emphasizes the need to evaluate **susceptibility** (not just single-turn correctness) under structured conversational influence.

## How is this different from GALILEO?

- Focuses on **multimodal persuasion** and LVLMs (vision-language) rather than primarily text-only multi-turn robustness/sycophancy/belief revision controls.
- Evaluates persuasion in contexts like advertising and misinformation with theory-based strategies, rather than GALILEO’s core emphasis on multi-turn robustness under pressure and drift control mechanisms.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO focuses on robustness, belief revision norms, and anti-sycophancy under iterative prompting, it may provide **cleaner causal tests** for *instructional pressure* vs *evidence-driven updates* in text-only dialogues.
- GALILEO can potentially better separate “legitimate revision” from “social/pressure compliance” (depending on its protocol and labels).

## Where GALILEO is weaker / needs to improve

- If GALILEO is text-only, it likely undercovers a major real-world channel: **visual persuasion** (memes, infographics, screenshots, short videos).
- GALILEO may need stronger evaluation coverage for **misinformation persuasion** that exploits multimodal cues.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add (or at least discuss) a **multimodal extension**: run a small-scale ablation with text-only vs captioned vs image/video evidence/suggestions to quantify additional susceptibility.
- [ ] Consider adopting/analogizing **trajectory metrics** like PDCG: reward early/strong drift vs stability across turns.
- [ ] Include a related-work paragraph framing multimodal persuasion as an adjacent threat model for multi-turn robustness and preference consistency.
- [ ] If feasible, add a “prior preference / stubbornness” manipulation in GALILEO-style tasks to test whether pressure/evidence overrides stated preferences.

## Quotes / details to potentially cite

- Dataset scale claim: “450 scenarios, 62,160 images, 4,756 videos” (as described in the paper’s introduction).
- Reported qualitative insights (RQ summaries): multimodal increases persuasion (esp. misinformation); prior preferences reduce susceptibility but multimodal cushions; strategy effectiveness depends on context (reciprocity vs credibility/logic).
