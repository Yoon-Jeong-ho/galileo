# Linear Representations of Political Perspective Emerge in Large Language Models

- Year: 2025
- Venue: ICLR 2025 (conference paper; arXiv)
- Authors: Junsol Kim et al.
- URL: https://arxiv.org/abs/2503.02080
- BibTeX key (if we add it): kim2025linear_political_perspective_llms
- Tags: representations, political-stance, probing, mechanistic-interpretability, steering, stability

## One-sentence takeaway

Political ideology (liberal–conservative axis) is linearly decodable from a small set of attention-head activations in several open LLMs and can be causally steered via linear interventions on those heads.

## What problem does it solve?

- Mechanistically characterizes *where/how* political perspective is represented inside LLMs, beyond prompt-level “bias tests”.
- Provides a monitoring signal for the implicit political stance an LLM is adopting during open-ended generation.
- Demonstrates a concrete steering mechanism (activation intervention) to shift stance without prompt engineering / finetuning.

## What is the core method / protocol?

- Models: Llama-2-7b-chat, Mistral-7b-instruct, Vicuna-7b.
- Data/protocol (as described in abstract/intro):
  - Prompt the LLM to write essays “from the perspective of” different U.S. lawmakers.
  - Use each lawmaker’s DW-NOMINATE score as a scalar ideology target label.
  - Train linear probes on attention-head activations across layers; identify heads/sets of heads that best predict ideology.
  - Show transfer: probes trained on lawmakers predict political slant of generations when prompted as news outlets.
  - Use interventions on identified heads (a steering vector with strength parameter \alpha) to make text more liberal vs conservative.

## What are the key metrics?

- Predictive performance of probes on held-out lawmakers (reported as correlations in the paper; intro mentions Spearman).
- “Do we need nonlinearity?” comparison: linear vs non-linear probes (claimed no gain from non-linear).
- For steering: correlation between intervention strength \alpha and perceived political slant; plus per-issue breakdown.

## What are the main results?

- A small subset of attention heads yields strong linear predictability of DW-NOMINATE; most predictive heads are in middle layers.
- Probes generalize from lawmakers to news outlets (predicting established outlet-slant measures).
- Linear interventions on these heads steer generated text along the ideology axis; effectiveness varies by model and increases with more heads intervened on.
- Base (no-intervention) generations trend slightly liberal on their 1–7 slant scale (per intro).

## How is this similar to GALILEO?

- Shares the theme of *latent state / stance* as an internal variable that can be monitored (probe) and manipulated (intervention).
- Useful as related work for “representation + monitoring” framing when discussing drift/instability of subjective perspectives.

## How is this different from GALILEO?

- Focuses specifically on U.S. political ideology and mechanistic interpretability (attention-head probing + activation steering), not general belief/stance robustness across tasks.
- Primary evaluation is probe predictability and steering efficacy, rather than longitudinal multi-turn stability or resilience to adversarial pressures (if those are central to GALILEO).

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO centers on *behavioral* robustness (multi-turn drift, recovery, evidence-based revision controls), it can offer broader, task-agnostic protocols than a single ideology axis.

## Where GALILEO is weaker / needs to improve

- If we lack mechanistic monitoring/steering hooks, this paper is a strong example of turning a “stance” concept into a measurable internal signal + causal lever.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding a “latent-stance probe” baseline: train linear probes for an axis relevant to GALILEO (not necessarily political), then track probe value over turns as a drift metric.
- [ ] Consider a causal test: identify a small set of internal components correlated with the stance variable and test whether interventions change downstream behavioral metrics (with controls).
- [ ] Related-work writeup angle: contrast prompt-based bias/stability tests with mechanistic monitoring + steering.

## Quotes / details to potentially cite

- Abstract: they “identify sets of attention heads whose activations linearly predict … DW-NOMINATE scores”.
- Abstract: predictive heads are “primarily located in the middle layers”.
- Abstract: “applying linear interventions to these attention heads” can steer outputs liberal vs conservative.
