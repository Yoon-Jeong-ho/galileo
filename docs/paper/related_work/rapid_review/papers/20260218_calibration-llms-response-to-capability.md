# On Calibration of Large Language Models: From Response To Capability

- Year: 2026
- Venue: arXiv (preprint)
- Authors: Sin-Han Yang; Cheng-Kuang Wu; Chieh-Yen Lin; Yun-Nung Chen; Hung-yi Lee; Shao-Hua Sun
- URL: https://arxiv.org/abs/2602.13540
- BibTeX key (if we add it): yang2026calibration-capability
- Tags: calibration, confidence, evaluation, pass@k

## One-sentence takeaway

They argue that “response calibration” (confidence for one sampled output) is misaligned for stochastic decoding, and propose “capability calibration” to estimate a model’s expected per-query success probability, improving pass@k prediction and inference-budget allocation.

## What problem does it solve?

- When LLMs are sampled/decoded stochastically, the correctness of one generated response is a noisy proxy for whether the model can solve the query at all.
- Many downstream decisions (retrying, allocating more samples, routing, abstaining) depend on *capability* (expected chance of success), not on the probability that *this particular* response is correct.

## What is the core method / protocol?

- Define and distinguish:
  - **Response calibration**: confidence attached to a specific generated response.
  - **Capability calibration**: confidence targeting the model’s *expected accuracy on the query* under the decoding distribution.
- Provide theoretical/empirical evidence that the two notions differ under stochastic decoding.
- Empirical setup evaluates multiple confidence estimation methods for how well they predict capability-related targets (including pass@k).
- Demonstrate that capability-calibrated confidence can be used to:
  - better predict **pass@k**
  - better allocate inference budget (e.g., decide when to draw more samples)

(Details of specific estimators are not fully available from the abstract-only skim; revisit PDF if we need estimator-specific citations.)

## What are the key metrics?

- Calibration quality for capability vs response (likely ECE/Brier-style; not specified in abstract)
- **pass@k prediction** quality
- Utility for **inference budget allocation** (e.g., accuracy vs compute tradeoff)

## What are the main results?

- Capability and response calibration differ “theoretically and empirically”.
- Capability-calibrated confidence improves:
  - pass@k prediction
  - inference-budget allocation decisions

## How is this similar to GALILEO?

- Aligns confidence/uncertainty with the *decision objective* (system-level performance under compute constraints), rather than just per-output correctness.
- Emphasizes evaluation protocols that match deployment settings (multi-sample decoding, variable budgets).

## How is this different from GALILEO?

- Primarily a calibration framing + evaluation for LLM confidence under stochastic decoding, rather than a new end-to-end controller/agentic method (based on abstract).
- Focus is on *confidence estimation targets* (capability) and their calibration, not necessarily on broader system orchestration.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO already models multi-try / budgeted inference explicitly, it can incorporate capability-calibrated signals as one component but still optimize the whole pipeline.

## Where GALILEO is weaker / needs to improve

- If our current writeup/evals discuss calibration only at the response level, we may be misaligned with stochastic decoding and pass@k-style objectives.

## Action items for GALILEO (experiments / method / writing)

- [ ] In related work, explicitly distinguish response-level confidence vs capability (per-query success) under stochastic decoding.
- [ ] Add an evaluation slice: confidence → predict pass@k and/or choose k adaptively; report budget–accuracy tradeoff.
- [ ] Check whether any of our confidence baselines should be reinterpreted/retargeted to capability calibration.

## Quotes / details to potentially cite

- Abstract framing: prior work focuses on “response-level confidence” but practical settings ask “how likely a model is to solve a query overall”.
- Proposed term: “capability calibration… targets the model’s expected accuracy on a query”.
- Claimed impact: “capability-calibrated confidence improves pass@k prediction and inference budget allocation.”
