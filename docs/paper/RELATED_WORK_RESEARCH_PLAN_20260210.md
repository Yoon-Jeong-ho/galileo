# Related-work research plan (2026-02-10)

Goal: strengthen GALILEO’s **methodology + positioning** (not just add citations) by explicitly borrowing/contrasting with the strongest *multi-turn* evaluation paradigms.

## A) Candidate papers to integrate (high priority)

### A1) Time-To-Inconsistency: A Survival Analysis of LLM Robustness to Adversarial Attacks (Li et al., 2025/2026)
- URL: https://arxiv.org/abs/2510.02712
- What it contributes:
  - Formalizes conversational robustness as a **time-to-event** process and applies survival analysis (Cox/AFT/RSF).
  - Uses drift features; shows hazard spikes with abrupt drift; proposes lightweight risk monitor.
- How it changes *our* paper:
  - Strengthens our methodological framing: TOF + survival curves are a special case of time-to-event.
  - We can add a short paragraph in Method/Related Work: “we keep survival curves as direct, interpretable metrics on *ground-truth tasks*, rather than fitting parametric hazard models; but survival-analysis framing supports our design.”
  - Optional analysis add-on (if we want): fit an AFT or Cox model on our per-turn correctness logs with persona/control indicators.
- Action item:
  - Add 2–3 sentences to Sec 6.4 positioning + cite as survival-analysis neighbor.

### A2) ReviseQA: A Benchmark for Belief Revision in Multi-Turn Logical Reasoning
- URL: https://openreview.net/pdf?id=Z4KBiAYXlI
- What it contributes (from snippet/metadata):
  - Explicit belief revision / sequential “edit turns” tasks (multi-turn logical reasoning).
- How it changes *our* paper:
  - Helps us distinguish **“should update belief when new evidence arrives”** vs **“should resist social pressure without evidence”**.
  - We can use this to justify the Neutral Re-asking Control as a drift baseline and clarify that our personas are *pressure without new ground-truth evidence*.
- Action item:
  - Write 1 positioning paragraph contrasting “evidence-based revision” vs “pressure-induced flip” and cite ReviseQA.

### A3) When Two LLMs Debate, Both Think They’ll Win (Nguyen et al., 2025)
- URL: https://arxiv.org/abs/2505.19184v3
- What it contributes:
  - Multi-turn belief updates + confidence dynamics; shows systematic overconfidence and escalation.
- How it changes *our* paper:
  - Supports motivation: multi-turn interaction can produce pathological dynamics even with repeated feedback.
  - Could inspire a lightweight **confidence proxy** analysis (optional) if we can log something stable.
- Action item:
  - Add 1–2 motivation sentences in Intro (multi-turn belief updates are nontrivial; known pathologies in debate/confidence settings).

## B) Concrete integration plan (paper-writing lane)

1) Update `docs/paper/PAPER_DRAFT_EN.md` Sec 6.4 with a **tight survival-analysis positioning** paragraph (Time-To-Inconsistency).
2) Add a short “belief revision vs pressure” clarification paragraph (ReviseQA) in either Intro gap or Related Work.
3) Add 1 motivation sentence (debate confidence escalation) in Intro or Related Work.

## C) Optional paper-facing experiments/analysis (later)

- Fit a simple survival model (Cox/AFT) on our logs with covariates: persona/control, task family, round, etc.
- Produce a small table: hazard ratio (persona vs control), with a cautionary note about assumptions.

## D) Next query batch (if we want more neighbors)

- Multi-turn instruction following endurance benchmarks (EvolIF, etc.)
- Consistency/limit-awareness under uncertainty benchmarks (CAR-bench)
