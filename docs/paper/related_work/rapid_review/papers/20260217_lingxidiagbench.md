# LingxiDiagBench: A Multi-Agent Framework for Benchmarking LLMs in Chinese Psychiatric Consultation and Diagnosis

- Year: 2026
- Venue: arXiv
- Authors: Shihao Xu, Tiancheng Zhou, Jiatong Ma, Yanli Ding, Yiming Yan, Ming Xiao, Guoyi Li, Haiyang Geng, Yunyun Han, Jianhua Chen, Yafeng Deng
- URL: https://arxiv.org/abs/2602.09379
- BibTeX key (if we add it): lingxidiagbench_xu_2026
- Tags: multi-agent, clinical, benchmark, multi-turn, diagnosis, patient-simulation, chinese

## One-sentence takeaway

A large-scale Chinese psychiatric consultation benchmark finds that LLM diagnostic performance drops sharply when moving from simple static classification to comorbidity recognition and 12-way differential diagnosis, and that “good-looking” consultations (LLM-judge) only moderately correlate with diagnostic correctness.

## What problem does it solve?

- Lack of a benchmark that jointly supports (i) realistic patient simulation, (ii) clinician-verified labels (or label process aligned to clinical categories), and (iii) *dynamic multi-turn* psychiatric consultation/diagnosis evaluation—here, specifically for Chinese.
- Provides a setting to test whether an LLM can *ask the right questions* (information gathering) and then make the right diagnosis, rather than only answering a fixed prompt.

## What is the core method / protocol?

- Introduces **LingxiDiagBench**, a **multi-agent** evaluation framework with two modes:
  - **Static diagnostic inference** (given a case / information, produce diagnosis).
  - **Dynamic multi-turn consultation** (conduct an interview-like dialogue; then diagnose).
- Releases **LingxiDiag-16K**: 16,000 EMR-aligned synthetic consultation dialogues aiming to match real clinical demographic/diagnostic distributions across **12 ICD-10 psychiatric categories**.

## What are the key metrics?

- Diagnostic accuracy in different regimes:
  - Binary depression–anxiety classification.
  - Depression–anxiety **comorbidity recognition**.
  - **12-way differential diagnosis** accuracy.
- “Consultation quality” scored by **LLM-as-a-Judge** (and correlation with diagnostic accuracy).

## What are the main results?

- Strong performance on easy binary classification (reported up to **92.3%** for depression vs anxiety).
- Large degradation on harder clinical distinctions:
  - Comorbidity recognition: **43.0%**.
  - 12-way differential diagnosis: **28.5%**.
- **Dynamic consultation underperforms static evaluation**, suggesting that weaknesses in *information-gathering strategies* hurt downstream diagnostic reasoning.
- LLM-judge consultation quality shows **only moderate correlation** with diagnostic accuracy: asking structured questions is not sufficient for correct diagnosis.

## How is this similar to GALILEO?

- Multi-turn setting where *interaction dynamics* (questioning / follow-ups) can induce failures that static evaluation misses.
- Highlights a key evaluation pitfall: **proxy metrics** (e.g., judge-rated “quality”) may not track the *true target outcome* (correctness), echoing concerns about evaluator brittleness / misalignment.

## How is this different from GALILEO?

- Domain: psychiatric consultation/diagnosis (clinical) vs GALILEO’s focus on multi-turn robustness under pressure / drift vs revision.
- Failure modes: poor information acquisition and differential diagnosis complexity rather than persuasion-induced drift or stance flips.
- Metrics center on accuracy by diagnostic granularity, not time-to-event / survival / flip dynamics.

## Where GALILEO is stronger / cleaner (if true)

- GALILEO can provide clearer *mechanistic attribution* (pressure-driven drift vs evidence-driven revision controls) and time-to-failure style metrics; LingxiDiagBench (per abstract) is more domain/task-centric.

## Where GALILEO is weaker / needs to improve

- If GALILEO includes any interview-like / agentic components, this paper is a reminder to evaluate **information gathering** explicitly, not just final answers.
- Judge-based quality scoring is risky; GALILEO should validate judge proxies against ground-truth outcomes wherever possible.

## Action items for GALILEO (experiments / method / writing)

- [ ] In any agentic/multi-turn slice, separately score **(a) info-gathering quality** and **(b) downstream correctness**, and report their correlation (don’t assume they align).
- [ ] Add a brief related-work note: “multi-turn dynamic evaluation can underperform static evaluation due to poor questioning/information acquisition; judge-rated dialogue quality may only weakly predict correctness.”

## Quotes / details to potentially cite

- “LLMs achieve high accuracy on binary depression--anxiety classification (up to 92.3%), [but] performance deteriorates substantially for depression--anxiety comorbidity recognition (43.0%) and 12-way differential diagnosis (28.5%).”
- “Dynamic consultation often underperforms static evaluation, indicating that ineffective information-gathering strategies significantly impair downstream diagnostic reasoning.”
- “Consultation quality assessed by LLM-as-a-Judge shows only moderate correlation with diagnostic accuracy …”
