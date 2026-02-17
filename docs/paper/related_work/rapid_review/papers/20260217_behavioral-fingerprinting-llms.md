# Behavioral Fingerprinting of Large Language Models

- Slug: behavioral-fingerprinting-llms
- Year: 2025
- Venue: Agents4Science 2025 (submission; arXiv)
- Authors: Zehua Pei; Hui-Ling Zhen; Ying Zhang; Zhiyuan Yang; Xing Li; Xianzhi Yu; Mingxuan Yuan; Bei Yu
- Links:
  - paper: https://arxiv.org/abs/2509.04504
  - project/code (if any): https://github.com/JarvisPei/Behavioral-Fingerprinting
- Bibtex: https://arxiv.org/abs/2509.04504 (see arXiv for bib)

## 1) What problem does it study?
How to *differentiate* LLMs beyond single benchmark scores by characterizing their “intrinsic cognitive and interactive styles”. The paper proposes a **behavioral fingerprint**: a multi-axis diagnostic profile intended to reveal where models with similar capability scores behave differently (e.g., sycophancy and robustness).

## 2) Experimental setup (what is being measured?)
- Task(s): a **Diagnostic Prompt Suite** of 21 prompts grouped into four categories:
  1) Internal “world model” (e.g., counterfactual physics)
  2) Reasoning & cognitive abilities (e.g., abstract pattern finding)
  3) Biases & personality (includes **sycophancy** + an MBTI-analogue communication-style classification)
  4) Robustness (semantic consistency under paraphrase)
- Perturbation/pressure type:
  - **Sycophancy probe**: present a factually incorrect user premise and see whether the model corrects vs “plays along” (example given: “As we know, the Earth is flat… plan a flight path… to the ice wall”).
  - **Robustness probe**: paired prompts with the same meaning but different phrasing; score semantic consistency.
- Multi-turn? N (appears primarily single-turn diagnostics; no interaction trajectories).
- Metrics:
  - An automated **LLM-as-judge** scoring pipeline: a strong evaluator model (they state Claude-opus-4.1) scores each response with a prompt-specific rubric.
  - Scores aggregated by category and normalized to **[0,1]**, visualized as radar “fingerprints” + comparative bar charts.
  - “Personality” reported as an MBTI-analogue label derived from judged communication style.

## 3) Key findings (bullet)
- **Convergence**: frontier models show strong convergence on core reasoning capabilities (abstract reasoning, causal-chain reasoning).
- **Divergence**: alignment-related traits vary widely even among frontier models:
  - **Sycophancy resistance** (large-model group) ranges roughly from **1.00** (high resistance; e.g., Claude-opus-4.1, LLaMA-3.1-405B-Instruct) down to **0.25** (low resistance; e.g., Grok-4).
  - **Robustness** (semantic consistency) ranges from **1.00** down to **0.50** for large models.
- “Default persona” clustering: many models cluster around ISTJ/ESTJ-like profiles, hypothesized to reflect shared alignment incentives.

## 4) Limitations / threats
- Judge dependence: results depend on the chosen **evaluator LLM** and rubric wording; may embed evaluator biases.
- Not multi-turn: does not measure **time-to-failure**, persistence, oscillation, or recovery dynamics.
- Sycophancy is probed via a *small diagnostic set*; unclear how robust the ranking is across topics/pressures.
- Robustness definition is paraphrase-consistency-focused; may not capture belief drift under social pressure.

## 5) How it relates to GALILEO
- What we can cite it for:
  - Evidence that **sycophancy resistance varies dramatically** across top models even when “capability” is similar.
  - A clean framing: as task accuracy saturates, *interactive/alignment behavior becomes a key axis of differentiation*.
  - An example of scalable, rubric-based evaluation via **LLM-as-judge** (with reproducibility claims).
- Where we differ (our delta):
  - GALILEO is about **multi-turn** robustness under social pressure, with explicit controls and trajectory metrics (flip timing, persistence, recovery), whereas this is primarily **single-turn diagnostics**.
- Direct mapping:
  - Survival ↔ not present (no censoring/time-to-event)
  - TOF ↔ not present
  - Recovery ↔ not present
  - Neutral Re-asking Control ↔ not present

## 6) Quote-able lines
- Paraphrase targets:
  - “As performance on core tasks converges, evaluation should ask not only *is it correct* but *how does it think*?”
  - “Alignment-related behaviors such as sycophancy and semantic robustness vary dramatically across frontier models.”

## 7) Actions
- [ ] Add to paper: related-work paragraph motivating why we need *behavioral* metrics beyond accuracy, citing their observed sycophancy/robustness divergence.
- [ ] Add to bib
