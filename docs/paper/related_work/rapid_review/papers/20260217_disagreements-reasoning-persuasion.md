# Disagreements in Reasoning: How a Model’s Thinking Process Dictates Persuasion in Multi-Agent Systems

- Year: 2025
- Venue: arXiv (work in progress)
- Authors: Haodong Zhao, Jidong Li, Zhaomin Wu, Tianjie Ju, Zhuosheng Zhang, Bingsheng He, Gongshen Liu
- URL: https://arxiv.org/abs/2509.21054
- BibTeX key (if we add it): Zhao2025DisagreementsReasoningPersuasion
- Tags: persuasion, multi-agent, reasoning, chain-of-thought, robustness

## One-sentence takeaway

Explicit reasoning (“thinking mode”) makes models harder to persuade, but *sharing* that thinking content can substantially increase their ability to persuade other agents.

## What problem does it solve?

- As LLM-based multi-agent systems (MAS) become common, we need to understand **persuasion dynamics between agents** (not just human↔LLM).
- Challenges the “bigger models persuade better” assumption; argues persuasion is shaped by **cognitive process / explicit reasoning**.

## What is the core method / protocol?

- **Pairwise persuasion games** between model pairs, with roles: persuader and persuadee.
- Evaluate both:
  - **Objective** questions: MMLU multiple-choice; they standardize correct answer to option A and set persuasion target to option D.
  - **Subjective** questions: 1k claims sampled from PersuasionBench and Perspectrum; stances mapped to {support, neutral, oppose} with targets set to flip stance (details depend on initial stance).
- Key manipulation for LRMs: whether the persuader includes its **thinking content** (e.g., content inside `<think>...</think>`) in the message shown to the persuadee.
- They report **model-by-model heatmaps** of persuasion success across pairs (row = persuader, column = persuadee) for objective and subjective datasets.

## What are the key metrics?

Defined on an evaluation subset where the persuadee is initially correct (or aligned with a baseline answer):

- **Persuaded-Rate (PR)**: fraction where persuadee’s post-persuasion answer equals the persuasion target.
- **Remain-Rate (RR)**: fraction where persuadee keeps the initial answer.
- **Other-Rate (OR)**: fraction where persuadee changes, but not to the target.

(They define these formally and compute via indicator functions over the filtered set.)

## What are the main results?

- **Weaker models are more persuadable** (higher PR as persuadees), but **persuader capability has weaker effects** on PR than persuadee weakness.
- **Subjective questions** show higher persuadability than objective ones.
- **Sharing thinking content boosts persuasiveness of LRMs** as persuaders (reported average PR increase ~21% in one objective setting).
- **Thinking mode for persuadees** tends to increase resistance to persuasion (lower PR when thinking mode enabled as persuadee), though effects for persuaders are mixed.

## How is this similar to GALILEO?

- Same underlying theme: **robustness of model beliefs/answers under social pressure / persuasive interaction** across multiple turns/agents.
- Highlights that “stronger reasoning” can reduce susceptibility, aligning with GALILEO’s interest in separating *capability* from *robustness under pressure*.

## How is this different from GALILEO?

- Focuses on **agent↔agent persuasion**, not a controlled “pressure vs evidence” belief-revision design.
- Primary outcome is **flip-to-target rate** (PR/RR/OR) rather than time-to-event / survival-style turn-of-failure metrics or explicit recovery trajectories.
- Uses **exposing internal reasoning content** as a treatment variable; GALILEO may prefer protocols that don’t depend on access to hidden reasoning traces (or treat it as optional).

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO includes paired controls for evidence-driven updates vs pressure-only drift, that is cleaner than PR-only flips.
- If GALILEO reports time-to-failure and recovery-after-flip, it can capture dynamics that PR aggregates away.

## Where GALILEO is weaker / needs to improve

- This paper suggests a potentially large effect from **reasoning transparency** on persuasion outcomes; GALILEO should consider whether “show your reasoning” policies are a confound or a controllable experimental factor.
- GALILEO may want an explicit **multi-agent** slice (debate / committee / chain persuasion), not only single-user pressure.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add an ablation: pressure dialogues **with vs without exposed reasoning** (or with a “rationale sharing” channel) to test whether transparency increases persuader strength while also affecting susceptibility.
- [ ] When discussing persuasion safety in agentic settings, cite the **Persuasion Duality** framing: reasoning can harden an agent against persuasion but make its persuasion attempts stronger when shared.
- [ ] Consider extending to **multi-hop** settings (influence propagation/decay) if GALILEO claims cover multi-agent ecosystems.

## Quotes / details to potentially cite

- They argue persuasion is “fundamentally dictated by a model’s underlying cognitive process, especially its capacity for explicit reasoning,” not just scale.
- “Adding thinking content significantly boosts the persuasiveness of LRMs as persuaders” (reported average PR gain ~21% on an objective setting).
