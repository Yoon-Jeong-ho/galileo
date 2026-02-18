# Aligning Dialogue Agents with Global Feedback via Large Language Model Multimodal Reward Decomposition

- Year: 2025
- Venue: arXiv
- Authors: Dong Won Lee; Hae Won Park; Cynthia Breazeal; Louis-Philippe Morency
- URL: https://arxiv.org/html/2505.15922
- BibTeX key (if we add it): lee2025llmgeli
- Tags: dialogue-agents, rlhf, reward-decomposition, temporal-credit-assignment, multimodal, preference-learning

## One-sentence takeaway

Uses a frozen LLM to decompose a single session-level conversation rating into turn-level pseudo-rewards (optionally using multimodal listener cues), then distills those into a text-only reward model for RL fine-tuning of dialogue agents.

## What problem does it solve?

- In long-form dialogue, realistic human feedback is often *global* (session-level) rather than dense turn-by-turn labels, making credit assignment for RLHF hard.
- Prior reward redistribution / decomposition methods either require manual reward shaping or struggle to use rich social signals.

## What is the core method / protocol?

- Two-stage pipeline:
  1) **Global Explicit (GE) reward decomposition**: prompt a *frozen* pretrained LLM with (i) the full transcript and (ii) the final session score, and ask it to assign per-turn contributions that (softly) sum to the session score.
     - **Text-only variant (LLM-GELI)**: decomposition uses only the transcript.
     - **Multimodal variant (MM-LLM-GELI)**: additionally conditions on *listener* behavioral cues (pitch, gaze, facial affect, etc.) that are converted into natural-language descriptors aligned to each utterance.
  2) **Local Implicit (LI) reward modeling**: train a lightweight *text-only* reward model on state-action pairs to regress to the LLM-produced per-turn pseudo-rewards (MSE loss). This distills multimodal/LLM credit assignment into a deployable text-only scorer.
- Then perform RL fine-tuning (PPO + KL regularization; LoRA) of a dialogue model (they use LLaMA-2) using the learned reward model.

## What are the key metrics?

- **Human evaluation** of generated responses on multiple dialogue-quality dimensions (9 criteria; pairwise/3-way comparisons).
- Reward-model diagnostics:
  - **Global Loss**: MSE between true session score and sum of predicted per-turn rewards.
  - **Local Difference**: difference in predicted reward when listener visual affect is positive vs not (tests alignment to local implicit cues).

## What are the main results?

- On CANDOR (long-form video conversations with post-session survey ratings), LLM-based decomposition reduces global-loss error vs classical return decomposition baselines (RUDDER / IRCR / RRD) and a prior dialogue-focused method (GELI).
- Multimodal prompting improves human-evaluated social/affective qualities compared to text-only decomposition.
- Shows some out-of-distribution transfer (evaluations reported on SODA; also ESConv for emotional support dialogue).

## How is this similar to GALILEO?

- Directly targets **trajectory-level feedback → local supervision** conversion, which is the same bottleneck many agent-alignment / evaluation setups face.
- Treats an LLM as a **judge / interpreter** that produces structured supervision signals used to train smaller, task-specific components.

## How is this different from GALILEO?

- Focus is **dialogue quality alignment** with PPO, not general tool-using agents or multi-step task success.
- Uses an LLM primarily for **credit assignment / reward decomposition**, not necessarily for orchestrating actions or evaluation harness design.
- The multimodal component is handled by **textualizing** nonverbal cues, rather than end-to-end multimodal modeling.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO emphasizes rigorous, task-grounded evaluation and reproducible protocols, it may avoid a key weakness here: **prompt sensitivity** in decomposition and reliance on a particular strong LLM.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently uses only end-of-trajectory metrics, this paper is a concrete recipe for turning *global outcomes* into *step-level diagnostics*.
- If GALILEO does not incorporate rich “implicit” signals, this suggests a path: include additional side-channel signals (even if only as structured text features) to improve attribution.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a related-work paragraph positioning: “LLM as reward decomposer / credit assignment oracle” vs classical temporal credit assignment (RUDDER/RRD/IRCR) and prior dialogue GELI.
- [ ] Consider a GALILEO ablation: global scalar outcome + LLM-based step attribution → train a lightweight scorer; compare to heuristics.
- [ ] If GALILEO has any auxiliary traces (timestamps, tool errors, user sentiment proxies), test “textualized multimodal” conditioning as input to the decomposer.

## Quotes / details to potentially cite

- Problem framing (session-level feedback): “annotators provide feedback only at the session level, reflecting their overall impression of the entire interaction.”
- Method summary: “decompose Global Explicit feedback into Local Implicit turn-level supervision signals that can be used for RLHF.”
- Multimodal angle: incorporate “behavioral cues, such as pitch, gaze, and facial affect, expressed as natural language descriptions.”
