# Structured Prompting Enables More Robust Evaluation of Language Models

- Year: 2025
- Venue: arXiv
- Authors: Asad Aali; Muhammad Ahmed Mohsin; Vasiliki Bikia; Arnav Singhvi; Richard Gaus; Suhana Bedi; Hejie Cui; Miguel Fuentes; Alyssa Unell; Yifan Mai; Jordan Cahoon; Michael (Mike) Pfeffer; Roxana Daneshjou; Sanmi Koyejo; Emily Alsentzer; Christopher Potts; Nigam H. Shah; Akshay S. Chaudhari
- URL: https://arxiv.org/abs/2511.20836
- BibTeX key (if we add it): aali2025structured
- Tags: prompting, evaluation, robustness, benchmarking, HELM, DSPy, prompt-optimization

## One-sentence takeaway

Integrating structured/optimized prompting (DSPy) into HELM shows that fixed benchmark prompts can systematically *underestimate* model capability and even flip leaderboard rankings, so “robust evaluation” should approximate a per-model performance ceiling over prompt variants.

## What problem does it solve?

- Public LM leaderboards often evaluate each benchmark with a **single fixed prompt**, but prompt sensitivity differs by model and task.
- This can yield **unrepresentative performance estimates** (underestimation; unstable gaps; rank flips), which is bad for deployment decisions.

## What is the core method / protocol?

- Build a reproducible **DSPy + HELM** integration where HELM’s baseline prompt is transformed into structured prompt variants.
- Compare multiple prompting methods:
  - HELM baseline (fixed, zero-shot, no CoT)
  - DSPy “Zero-shot Predict” (structured but unoptimized)
  - DSPy **Zero-shot CoT** (explicit reasoning field)
  - **BFRS** (Bootstrap Few-shot with Random Search): bootstrap candidate demos then random-search few-shot sets
  - **MIPROv2**: jointly optimize instructions + demos via proposal model + Bayesian/TPE-style search
- Evaluate 4 frontier LMs across 7 benchmarks (general + medical), reporting how results change under prompt variants.

## What are the key metrics?

- Primary: benchmark-specific accuracy / exact match (and task-appropriate checks, e.g., within-range for some MedCalc-Bench items).
- Meta-evaluation angles emphasized in the paper:
  - **Average performance change** vs HELM baseline (ceiling approximation)
  - **Variance / sensitivity across prompts** (stability)
  - **Rank/leaderboard flips** under alternative prompting

## What are the main results?

- Without structured prompting (fixed baseline prompts), HELM can:
  - **Underestimate performance** by ~4% on average (as reported in abstract).
  - Show **more variable estimates across benchmarks** (reported as +2% std dev in abstract).
  - **Misrepresent gaps** such that rankings **flip on 3/7 benchmarks** (abstract).
  - Adding CoT-style reasoning reduces sensitivity to prompt design (smaller Δ across prompt variants).

## How is this similar to GALILEO?

- Shares the core motivation that **robustness claims are meaningless without stress-testing the evaluation protocol** (here: prompt stress-testing rather than multi-turn social pressure).
- Supports a GALILEO-adjacent narrative: “single configuration” evaluation (single prompt / single interaction pattern) can hide failure modes or misestimate capability.

## How is this different from GALILEO?

- Focuses on **single-turn benchmark prompting** and *capability ceiling under prompt variation*, not multi-turn belief dynamics (drift vs revision), time-to-failure, or recovery.
- Treats robustness mainly as **stability to prompt phrasing/structure**, not robustness to persuasive pressure / adversarial dialogue moves.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO explicitly decomposes pressure-only drift vs evidence-driven revision and tracks **trajectory/time-to-event**, it offers a more behaviorally grounded robustness construct than prompt-only ceiling approximation.

## Where GALILEO is weaker / needs to improve

- If GALILEO reports results under only one (or a small number of) prompt templates, this paper is a cautionary precedent: we should quantify **prompt sensitivity** or show invariance.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a small “**prompt sensitivity**” appendix experiment: re-run core GALILEO conditions under 2–4 prompt *structures* (e.g., direct answer vs structured fields; with/without explicit reasoning; third-person framing), and report deltas.
- [ ] In the paper narrative, explicitly argue that GALILEO’s findings are not an artifact of one prompt template; cite this work as evidence that fixed prompts can flip conclusions.
- [ ] Consider a “**ceiling approximation**” framing for any headline metric that could be prompt-optimized (careful: we may want *worst-case* under pressure rather than best-case under optimization).

## Quotes / details to potentially cite

- Abstract-level claims to cite:
  - “HELM underestimates LM performance (by 4% average).”
  - “Leaderboard rankings flip on 3/7 benchmarks.”
- Method terms to cite/define succinctly:
  - DSPy structured prompting; BFRS; MIPROv2; “performance ceiling via prompt-only changes.”
