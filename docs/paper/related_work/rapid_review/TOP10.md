# Rapid Review — Top 10 shortlist (rolling)

This is a rolling shortlist of the **10 most important adjacent papers**.

For each item, we maintain:
- What it contributes
- What it misses relative to GALILEO
- What we should borrow
- How we should change GALILEO (experiments/methods/claims)

> NOTE: This list is expected to change frequently until ~30–50 papers are read.

## Shortlist

1) **Measuring Sycophancy of Language Models in Multi-turn Dialogues** (Hong et al., Findings of EMNLP 2025)
   - Contributes: concrete multi-turn social-pressure protocol + two extremely usable metrics: **Turn of Flip (ToF)** (time-to-failure proxy) and **Number of Flips (NoF)** (instability/oscillation).
   - Misses vs GALILEO: does not explicitly separate *evidence-driven belief revision* from *pressure-driven drift*; limited focus on *recovery after flip*.
   - Borrow: ToF/NoF framing and third-person prompting baseline (reported up to 63.8% reduction in debate).
   - How to change GALILEO: map our metrics onto a ToF/NoF-style vocabulary (or add direct analogues) and highlight our additional controls + recovery measurements.

