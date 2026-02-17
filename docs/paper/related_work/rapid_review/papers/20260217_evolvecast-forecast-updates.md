# Do Language Models Update their Forecasts with New Information?

- Year: 2025 (arXiv v1; updated 2026-02)
- Venue: ICML (Machine Learning track; arXiv preprint)
- Authors: Zhangdie Yuan; Zifeng Ding; Andreas Vlachos
- URL: https://arxiv.org/abs/2509.23936
- BibTeX key (if we add it): Yuan2025EvolveCast
- Tags: belief-update, forecasting, belief-revision, evaluation, calibration

## One-sentence takeaway

EvolveCast evaluates whether LLMs *update probabilistic forecasts* in response to post-cutoff news by comparing model belief deltas to *human forecaster* deltas (Metaculus), finding LLMs are responsive but systematically **too conservative / inconsistent** and poorly calibrated.

## What problem does it solve?

- Most LLM forecasting eval is static (single-shot) and/or judged by eventual resolution, which fails to measure whether a model’s *belief state* evolves appropriately as new evidence arrives.
- Need a protocol + metrics to test belief revision dynamics without rewarding “oracle-like” overconfidence.

## What is the core method / protocol?

- Build **EvolveCast** from Metaculus binary questions with dense human probability trajectories.
- Create paired timepoints:
  - **T0**: question start date; elicit model probability p0.
  - **T1**: later date when a relevant news update appears; elicit model probability pt conditioned on dated snippet.
- Define update delta: Δp = pt − p0; compare against **human aggregate** delta Δh = ht − h0.
- News alignment:
  - detect candidate “new info” times via Metaculus comments;
  - retrieve related news within ~1 week window; choose best match by semantic similarity between comment and article.
- Evaluate both:
  - **Black-box** (verbalized probability on a discrete scale normalized to [0,1])
  - **White-box** (logit/token-probability-derived confidence for the generated answer)
- Ablations:
  - “Single update” vs “Accumulated updates” (concatenate multiple snippets)
  - Direct “Up/Down/Still” directional QA prompting
  - Provide human reference forecast as context (no clear gains)

## What are the key metrics?

- **Directional agreement**: Mean Directional Accuracy (MDA) + precision/recall/F1 over {Up, Down, Still} after thresholding tiny deltas.
- **Magnitude alignment**: MSE over Δp vs Δh; plus a symmetric percentage error variant (SMSPE).
- **Calibration vs reference**: Brier-style squared error against human aggregate before/after; report change ΔBrier (negative better).

## What are the main results?

- LLMs do update in the *right direction sometimes*, but are often **under-reactive** (conservative deltas) or inconsistent relative to human reference.
- Neither verbalized nor logit-based confidence is consistently better; both remain far from human reference calibration.
- Reasoning-tuned variants (evaluated in the paper on DeepSeek-R1 distill models vs matched base backbones) improve over base models but still show conservatism.
- Adding more historical snippets (“accumulated news”) does **not** reliably help and can worsen directional agreement (likely dilution/noise / poor evidence weighting).
- If you directly ask for direction (Up/Down/Still), models do better than when you difference two numeric probabilities—suggesting direction reasoning is easier than calibrated magnitude updates.

## How is this similar to GALILEO?

- Same high-level target: **multi-turn belief dynamics** (how beliefs shift under new inputs) rather than one-shot accuracy.
- Emphasizes evaluating *process* (belief update) instead of final correctness, aligning with GALILEO’s “don’t reward the wrong failure mode” philosophy.

## How is this different from GALILEO?

- EvolveCast is **evidence-driven** updating with real-world “news” snippets and human forecaster references; not primarily about social pressure / persuasion attacks.
- Uses *human aggregate deltas* as the main reference target (Δh), rather than ground-truth resolution or correctness labels.
- Focuses on forecasting questions (Metaculus) and calibration-to-human, not (as far as visible from the core setup) recovery after adversarial pressure.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO includes explicit controls separating **pressure-only drift** vs **evidence-based revision**, it can make stronger causal claims about *why* belief changes occur (EvolveCast is about capability measurement, not pressure mechanism).
- If GALILEO reports recovery trajectories after induced failure, that’s beyond EvolveCast’s primary framing (which is delta alignment).

## Where GALILEO is weaker / needs to improve

- We should consider adopting EvolveCast’s “compare belief delta to a rational reference delta” concept for any GALILEO slice that involves genuine evidence updates.
- If we currently only report outcome flips / stability, we may be missing a clean **magnitude-of-update** story (under/overreaction) and calibration-to-reference.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a “delta-to-reference” evaluation mode for an evidence-update setting: compare model Δp to a reference Δ (human, market odds, or expert baseline) rather than to eventual outcomes.
- [ ] Report an “underreaction vs overreaction” decomposition (e.g., signed error of Δp−Δref) to separate stubbornness from gullibility.
- [ ] Consider a qualitative direction-only probe (Up/Down/Still) as a robustness diagnostic when numeric calibration is too noisy.
- [ ] If we use RAG/tool retrieval as an “update” mechanism, explicitly discuss the paper’s claim: models may treat retrieved text as *context* not *posterior evidence* → hence conservative updates.

## Quotes / details to potentially cite

- Framing: static evaluations miss “how forecasts and the confidence in them should evolve as new evidence emerges.”
- Key finding: updates are often “inconsistent or overly conservative” relative to human forecasters; confidence estimates (verbalized or logit-based) remain far from human reference.
- Methodological point: evaluate belief updates against **human aggregate** deltas rather than final resolution to avoid rewarding “clairvoyant” overconfidence.
