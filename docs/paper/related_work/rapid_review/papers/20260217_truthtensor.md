# TruthTensor: Evaluating LLMs through Human Imitation on Prediction Market under Drift and Holistic Reasoning

- Slug: truthtensor
- Year: 2026
- Venue: arXiv
- Authors: Shirin Shahabi; Spencer Graham; Haruna Isah
- Links:
  - paper: https://arxiv.org/abs/2601.13545
  - html: https://arxiv.org/html/2601.13545v3
  - project: https://truthtensor.com
  - code (if any): (not found in paper)
- Bibtex: https://doi.org/10.48550/arXiv.2601.13545

## 1) What problem does it study?
How to evaluate LLM “reasoning models” in settings with real-world uncertainty and distribution shift, where static benchmarks miss: (i) calibration, (ii) longitudinal drift, and (iii) human-aligned decision-making. The paper reframes evaluation as *human imitation* in live prediction markets, rather than only forecasting accuracy.

## 2) Experimental setup (what is being measured?)
- Task(s): probabilistic forecasting on live prediction markets (Polymarket), across 500+ real markets spanning political/economic/cultural/technological domains; forward-looking by construction.
- Perturbation/pressure type: time/streaming updates + changing market narratives; also token-budget constraints; (optional) “execution mode” trades based on an edge threshold.
- Multi-turn? Y (longitudinal re-querying over time; sampled at regular intervals such as daily until market resolution).
- Metrics:
  - Correctness: Brier score; log-likelihood; thresholded accuracy.
  - Calibration: ECE; MCE; reliability diagrams.
  - Drift: narrative drift score (reasoning trace inconsistency over time); temporal drift score (update appropriateness vs information arrival); confidence drift score (confidence/reasoning alignment); market divergence over time.
  - Risk/decision: VaR; CVaR; risk-adjusted returns.
  - “Economic” outcome: trading P&L (profit/loss) relative to market baseline (used as a behavioral signal for decision quality).
  - Efficiency: average input/output tokens.

## 3) Key findings (bullet)
- Static “forecast accuracy” can hide large differences: models with similar accuracy can diverge strongly in calibration, drift behavior, and risk sensitivity.
- Drift is presented as a first-class failure mode for long-horizon deployment: probability volatility and reasoning-trace divergence across time points can reveal instability that single-shot tests miss.
- The framework emphasizes contamination resistance: only forward-looking events + “instruction locking” (versioned, immutable prompt contracts) to make runs comparable.
- The paper reports a large-scale live evaluation window (Dec 12, 2025 – Jan 10, 2026) with very large volume (hundreds of thousands of decisions) and summarizes per-model aggregate signals including P&L and token usage.

## 4) Limitations / threats
- “Human imitation” target is proxied primarily via market-implied probabilities; markets can be biased/illiquid/manipulable, and not always normative.
- Drift metrics (esp. “reasoning trace divergence”) depend on access to a model’s reasoning traces and on the specific comparison method; reproducibility may hinge on proprietary logging choices.
- P&L as an evaluation signal can conflate modeling skill with market microstructure, liquidity, and execution assumptions.
- The paper is more of a paradigm/protocol than a tightly controlled benchmark; many degrees of freedom (sampling cadence, market selection, tool access, strategy thresholds).

## 5) How it relates to GALILEO
- What we can cite it for:
  - Motivation that *longitudinal instability/drift* matters and should be explicitly measured, not just single-turn accuracy.
  - A concrete example of “time-series querying until resolution” and reporting multiple axes (calibration + drift + efficiency).
  - Framing evaluation as behavior in dynamic, socially-grounded environments.
- Where we differ (our delta):
  - GALILEO focuses on *multi-turn robustness under adversarial/social pressure* (and recovery after flips) rather than forecasting markets.
  - GALILEO can treat the interlocutor/pressure as an explicit controlled variable; TruthTensor’s “pressure” comes from time + changing context.
- Direct mapping:
  - Survival ↔ time-to-degradation / longitudinal stability (their drift-over-time framing)
  - TOF ↔ first major divergence point vs baseline/market (not explicitly defined as “time-to-failure”)
  - Recovery ↔ not a core focus (no explicit “flip then recover” protocol)
  - Neutral Re-asking Control ↔ closest analog is repeated sampling with locked prompts and fixed cadence

## 6) Quote-able lines
- “TruthTensor asks: ‘How well does this model imitate human reasoning, calibration, and narrative coherence when confronted with evolving, uncertain, socially-grounded scenarios?’” (Intro)
- “TruthTensor places primary emphasis on measuring narrative drift, temporal inconsistency, and reasoning confidence decay…” (Key differentiators)

## 7) Actions
- [ ] Add to paper: related work section on *dynamic/forecasting-based evaluation* as an existence proof of drift-centric metrics and longitudinal querying.
- [ ] Add to bib
