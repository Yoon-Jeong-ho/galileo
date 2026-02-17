# On Robustness and Chain-of-Thought Consistency of RL-Finetuned VLMs

- Year: 2026
- Venue: arXiv
- Authors: Anshul Shah, Xiaoyu Zhu, Xinke Deng, Zhongyu Jiang, Yang Yang, Joerg Liebelt, Arnab Mondal (Apple)
- URL: https://arxiv.org/abs/2602.12506
- BibTeX key (if we add it): Shah2026RobustnessCoTConsistencyRLFtVLMs
- Tags: robustness, vision-language-models, chain-of-thought, faithfulness, RL-finetuning, calibration

## One-sentence takeaway

RL-finetuned “reasoning” VLMs gain benchmark accuracy but can become *more brittle* to simple misleading textual context and *less faithful/consistent* in their chain-of-thought, motivating robustness + faithfulness-aware evaluation (and rewards) beyond accuracy.

## What problem does it solve?

- The field is increasingly RL-post-training VLMs for visual reasoning, but typical evaluations emphasize **clean benchmark accuracy**.
- This paper stress-tests whether these RL-finetuned VLMs are robust to **benign textual perturbations** (that should not change the image-grounded answer) and whether their **CoT traces remain consistent/faithful**.

## What is the core method / protocol?

- Construct **controlled textual perturbations** on top of existing visual reasoning benchmarks (they mention 8 benchmarks targeting “simple” visual skills like counting / spatial relations).
- Two main perturbations:
  - **Wrong-Caption**: insert a misleading/incorrect caption assertion into the prompt.
  - **Wrong-Think**: seed the assistant with an **incorrect CoT prefix** that contains a misleading statement.
- Evaluate multiple open-source multimodal reasoning models and analyze:
  - accuracy drops under perturbations,
  - **CoT consistency** (whether the reasoning aligns with the final answer / with grounding),
  - entropy-based uncertainty shifts.
- Training-side analysis:
  - study RL fine-tuning dynamics,
  - test adversarial augmentation,
  - test a **faithfulness-aware reward**, and examine interactions/instabilities when combining with augmentation.

## What are the key metrics?

- Base accuracy vs perturbed accuracy (e.g., Wrong-Caption accuracy).
- CoT consistency / faithfulness indicators (described qualitatively; exact formalization likely in the paper).
- Entropy-based measures of uncertainty / probability mass shift on the correct option (calibration/robustness profile).

## What are the main results?

- Simple textual perturbations (misleading captions / wrong CoT prefixes) can cause **substantial robustness and confidence drops**.
- Effects are **more pronounced when accounting for CoT consistency** across models (i.e., models can output the right choice with inconsistent/unfaithful reasoning, or vice versa).
- They identify an **accuracy–faithfulness trade-off** during RL fine-tuning: more RL steps can increase accuracy while **eroding CoT reliability** and robustness to contextual shifts.
- Adversarial augmentation can improve robustness but **does not prevent faithfulness drift**.
- Faithfulness-aware reward can restore answer–reasoning alignment, but combining with augmentation can become unstable and still fail to deliver strong robustness.

## How is this similar to GALILEO?

- Like GALILEO, it argues that **headline accuracy can mask deeper failure modes** under controlled perturbations.
- The “wrong-context” perturbation idea is closely aligned with GALILEO-style stress testing: *hold the underlying task constant; vary context to reveal brittleness.*

## How is this different from GALILEO?

- Focuses on **vision-language reasoning** and **RL post-training dynamics** (faithfulness drift under RL), rather than multi-turn social pressure / belief revision (GALILEO’s likely core).
- Emphasis is on **text–vision modality conflict** and CoT faithfulness, not long-horizon dialogue trajectories.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO provides explicit controls separating *pressure-driven drift* from *evidence-driven revision*, it likely offers cleaner causal attribution than generic perturbation robustness.
- GALILEO’s multi-turn trajectory metrics (if central) can capture recovery/oscillation patterns beyond single-shot perturbation sensitivity.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently lacks a **faithfulness/consistency** axis (beyond answer correctness), this paper is a strong reminder that “reasoning traces” can degrade even as accuracy improves.
- If GALILEO touches multimodality, adding explicit **modality-conflict perturbations** could broaden the claim.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a “controlled misleading context” operator (analogous to Wrong-Caption / Wrong-Think) as an ablation/stressor, even in text-only settings (e.g., seeded wrong rationale).
- [ ] Add a “faithfulness/consistency” metric: cases where the model’s stated rationale contradicts its final answer or contradicts given evidence.
- [ ] In related work, cite the **accuracy–faithfulness trade-off under RL fine-tuning** as a cautionary point: post-training can improve benchmarks while worsening reliability under shifts.

## Quotes / details to potentially cite

- Abstract: “simple, controlled textual perturbations—misleading captions or incorrect chain-of-thought (CoT) traces—cause substantial drops in robustness and confidence…”
- Abstract: “uncover an accuracy-faithfulness trade-off: fine-tuning raises benchmark accuracy, but can simultaneously erode the reliability of the accompanying CoT…”
