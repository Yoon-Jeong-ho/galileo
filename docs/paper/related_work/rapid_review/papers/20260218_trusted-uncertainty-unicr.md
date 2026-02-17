# Trusted Uncertainty in Large Language Models: A Unified Framework for Confidence Calibration and Risk-Controlled Refusal

- Year: 2025
- Venue: arXiv
- Authors: Markus Oehri; Giulia Conti; Kaviraj Pather; Alexandre Rossi; Laia Serra; Adrian Parody; Rogvi Johannesen; Aviaja Petersen; Arben Krasniqi
- URL: https://arxiv.org/abs/2509.01455
- BibTeX key (if we add it): oehri2025trusted
- Tags: calibration, uncertainty, refusal, rag

## One-sentence takeaway

UniCR fuses multiple uncertainty signals into a calibrated probability of correctness and uses conformal risk control to decide when an LLM should refuse, improving risk–coverage tradeoffs without fine-tuning the base model.

## What problem does it solve?

- Deployed LLMs need a principled “answer vs refuse” decision, but raw token probabilities / entropy are often miscalibrated and do not reliably track semantic factual correctness (especially under distribution shift and for long-form generation).
- Existing abstention heuristics (entropy/logit thresholds, single-signal calibrators) often give worse coverage at a fixed error budget, and can be brittle across tasks (QA, code, RAG long-form).

## What is the core method / protocol?

- **Evidence fusion:** Collect heterogeneous “evidence” features that correlate with correctness, including:
  - sequence likelihood / log-prob features,
  - self-consistency / semantic dispersion across multiple samples,
  - retrieval compatibility signals (for RAG),
  - tool/verifier feedback (e.g., execution tests for code).
- **Calibration head:** Train a lightweight post-hoc mapping (temperature scaling + proper scoring rules) from evidence features to **P(correct)**.
  - Designed to work even for **API-only / black-box** LLMs (feature-based; no logits required beyond what API exposes).
- **Risk-controlled refusal:** Use **conformal risk control** to guarantee (distribution-free) an error/risk budget while maximizing coverage.
- **Long-form alignment:** Supervise confidence using **atomic factuality** signals derived from retrieved evidence to better track semantic fidelity (reduce “confident hallucinations”).

## What are the key metrics?

- Calibration: **ECE**, **Brier score**.
- Selective prediction / abstention: **risk–coverage curves**, **AURC** (area under risk–coverage curve), coverage at fixed risk.
- Task-specific correctness: short-form QA accuracy; code pass@k / execution test pass rate; long-form QA factuality/faithfulness measures (described as atomic factuality).

## What are the main results?

- Across short-form QA, code generation with execution tests, and RAG long-form QA, UniCR reports:
  - better calibration (lower ECE/Brier),
  - improved risk–coverage tradeoff (lower AURC),
  - higher coverage at the same target risk,
  compared to entropy/logit thresholds, post-hoc calibrators, and selective baselines.
- Qualitative/analysis claim: drivers of abstention are often **evidence contradiction**, **semantic dispersion**, and **tool inconsistency**, enabling more informative refusal rationales.

## How is this similar to GALILEO?

- Both care about **trustworthy deployment** decisions and evaluation under **risk/coverage** style objectives rather than only average accuracy.
- Uses RAG compatibility and factuality-derived supervision signals, which aligns with GALILEO’s concern about evidence-grounded behavior.

## How is this different from GALILEO?

- UniCR is primarily a **post-hoc calibration + selective decision** framework (probability-of-correctness + conformal risk control), not a new base model or retrieval method.
- Focus is on **refusal policy** and **calibration** across tasks (incl. code execution tools), rather than optimizing retrieval or generation quality directly.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO’s contributions are about model/system design beyond post-hoc calibration, GALILEO can claim a more **end-to-end** approach (UniCR explicitly avoids fine-tuning base LLM and relies on an added calibrator).

## Where GALILEO is weaker / needs to improve

- If GALILEO lacks a principled, user-controllable **risk budget** story (e.g., conformal risk control), this paper is a strong related-work anchor.
- If GALILEO uses simple confidence heuristics, UniCR suggests multi-signal fusion + calibration can materially improve risk–coverage.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a “**risk-controlled refusal**” baseline (conformal risk control / selective prediction framing) if we currently only have thresholding.
- [ ] Consider a **feature-fusion calibrator** for correctness using signals we already compute (retrieval agreement, self-consistency, verifier/tool checks) and report ECE/Brier + AURC.
- [ ] In related work, position our method vs: (1) entropy/logit thresholds, (2) post-hoc calibrators, (3) selective prediction methods, (4) conformal risk control for LLM outputs.

## Quotes / details to potentially cite

- “...turns heterogeneous uncertainty evidence—sequence likelihoods, self-consistency dispersion, retrieval compatibility, and tool/verifier feedback—into a calibrated probability of correctness and then enforces a user-specified error budget via principled refusal.”
- “...offers distribution-free guarantees using conformal risk control.”
- “...align confidence with semantic fidelity by supervising on atomic factuality scores derived from retrieved evidence...”
