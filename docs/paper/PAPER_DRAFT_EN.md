# GALILEO Paper Draft (EN) — Working scaffold (submission-oriented)

> Status: **English writing scaffold**. This file is meant to (i) lock down paper-quality phrasing and (ii) make the novelty/metrics unambiguous to reviewers.
> - Experiments/results/figures are tracked in the repo (`results/`, `paper_figures/`, `scripts/`).
> - BibTeX entries: `references.bib`.
> - Section numbers in headings are **placeholders** (we may renumber/remove them during LaTeX conversion).

---

## Abstract

Large language models (LLMs) can exhibit *belief-consistency failures* under social and rhetorical pressure—e.g., repeated denial, authoritative claims, or persuasive reframing—sometimes retracting previously correct answers even on tasks with clear ground truth. Prior work studies sycophancy and persuasion-related vulnerabilities (e.g., Sharma et al., 2025; Fanous et al., 2025; Huang et al., 2026), but a unified and reproducible protocol that measures **when** failures first occur (**turn-of-failure**), **how** robustness evolves over interaction rounds (**survival curves**), and **whether** models can recover after flipping (**recovery**) on ground-truth tasks remains limited.

We present **GALILEO**, a benchmark and pipeline for measuring multi-turn robustness under adversarial *persona-based* pressure. GALILEO (i) evaluates initial correctness, (ii) applies five adversarial personas for up to five rounds and quantifies **survival** as the probability an example remains correct through round *r* (a survival curve), and (iii) reports **turn-of-failure (TOF)** as the first round where a previously-correct example becomes incorrect, plus **recovery** rates on flipped examples. To ensure stable automatic scoring across tasks and multi-turn logs, we standardize the final answer format to `\boxed{...}` and perform boxed-first extraction in the evaluator. Initial snapshots indicate persona-dependent failure dynamics, sharper degradation on higher-uncertainty open-domain QA, and task-dependent recovery behavior.

---

## 1. Introduction

### 1.1 Motivation

In interactive settings, correctness is not a one-shot property. Even if an assistant produces a correct answer initially, subsequent conversation can apply pressure that nudges the model toward *agreeable* but wrong responses. This matters for high-stakes deployments (education, healthcare, legal assistance, research support), where a user may insist the model is wrong, cite “expert authority,” or repeatedly deny evidence.

A growing body of work investigates sycophancy (agreeing with user beliefs at the expense of truth) and persuasion-induced behavior change in LLMs (Sharma et al., 2025; Fanous et al., 2025; Huang et al., 2026), as well as instability under prompt/context variation (Tosato et al., 2025; Yu et al., 2026). However, evaluation protocols often (i) focus on single-turn outcomes or (ii) do not precisely characterize **the dynamics of failure across rounds** and **the possibility of recovery after a flip** on tasks with explicit ground truth.

### 1.2 Problem and evaluation gap

Single-turn accuracy does not answer:
- **When does a model first fail under pressure?** (early-turn vs late-turn failures)
- **How does robustness compound over rounds?** (survival curves)
- **Can a model recover after being misled?** (intervention/recovery)

We target a practically grounded setting: tasks with **ground-truth answers** where failure is unambiguous, while pressure is delivered through realistic conversational personas.

### 1.3 Contributions

1. **Ground-truth multi-turn dynamics.** We operationalize robustness under pressure as **survival curves** and **turn-of-failure**, and evaluate **recovery** in the same protocol.
2. **Unified multi-task pipeline.** We cover math, extractive QA, MCQA, and open-domain QA with a single runner/logging/evaluation interface.
3. **Stable evaluation via answer-format standardization.** We require a boxed final answer `\boxed{...}` for all tasks and use boxed-first extraction to reduce scoring ambiguity.
4. **Reproducibility and paper-ready exports.** We provide strict data directory construction, multi-seed aggregation (mean±std), and automated exports for tables/figures.
5. **Reviewer-facing controls and ablations.** We include controls that separate *persona pressure* from generic multi-turn drift (e.g., repeated denial without persona) and ablations that separate *robustness* from *recovery interventions*.

### 1.4 Core claims (and what must be shown in results)

**Narrative framing (paper through-line).** We frame the problem as a *betrayal of helpfulness*: alignment and preference-optimization can incentivize deference to user feedback, but in ground-truth domains this deference becomes a reliability failure (e.g., an assistant retracts a correct math answer after repeated denial). This motivates measuring **epistemic robustness**—the ability to maintain or return to truth under conversational pressure.

We structure the paper around three reviewer-checkable claims:

- **C1 (Dynamics):** Robustness under pressure is a *trajectory*, not a single number—single-turn accuracy misses *when* failures happen.
  - Evidence: persona-wise survival curves + TOF distribution.
- **C2 (Mechanism/structure):** Failure dynamics are systematically shaped by (i) *pressure mechanism* (persona) and (ii) *task uncertainty/evidence availability*.
  - Evidence: cross-task breakdowns (math vs extractive QA vs MCQA vs open QA) and uncertainty/answer-type stratification.
- **C3 (Intervention):** Recovery after flipping is measurable and *not equivalent* to robustness (being correct throughout); interventions can improve recovery without necessarily improving survival.
  - Evidence: recovery conditional on flip + recovery prompt ablations.

### 1.5 Minimum experiment set (submission-credible)

To make the above claims hard to dismiss, the camera-ready experimental core should include:

- **Multi-seed** (report mean±std) for the main model(s).
- **≥2 model families** under the same protocol (ideally 3 for stronger generalization).
- **Control condition**: a non-persona multi-turn baseline (e.g., generic denial/re-asking) to show effects are not purely conversational drift.
- **One intervention ablation**: at least one recovery prompt variant.
- **One sensitivity check**: decoding sensitivity (e.g., temperature sweep).

**Optional (if feasible): internal-state proxies.** If we can reliably extract token-level confidence signals (e.g., logit margin on the boxed answer token, entropy/uncertainty proxies), we will report *confidence decay* alongside behavioral flips. If not, we will treat uncertainty via task/answer-type stratification and consistency-based proxies, to avoid over-claiming.

### 1.6 Rebuttal prep (anticipated reviewer objections)

- *“Isn’t agreeing with the user just being helpful?”* Our focus is on **ground-truth domains** where deference to incorrect user feedback is a functional failure (education, medical triage, legal assistance). In these settings, “helpfulness” that abandons truth is miscalibrated behavior.
- *“Is this just long-context degradation / generic drift?”* We include a **Neutral Re-asking Control** (generic denial/re-asking) to separate persona-specific pressure mechanisms from multi-turn drift.

---

## 2. Task setting

We focus on tasks where ground-truth evaluation is feasible:
- **Math:** GSM8K, SVAMP (final numeric/string answer).
- **Extractive QA:** SQuAD-style questions (answer aliases).
- **MCQA:** ARC-Easy (A/B/C/D labels).
- **Open-domain QA:** TriviaQA (answer aliases; higher uncertainty).

Each dataset is stored as **JSONL**, with one example per line. We unify fields across tasks by storing at minimum:
- `task`: one of `{math, qa, mcqa}` (openQA is treated as `qa` with different datasets).
- `question`: the input.
- `ground_truth`: canonical answer (string) when applicable.
- `choices` and `label` for MCQA.

---

## 3. GALILEO protocol

Given a dataset and an LLM, GALILEO proceeds in three phases:

### Phase 1: Initial evaluation

We prompt the model to answer each question. We score the response against ground truth. Let `C` be the set of examples answered correctly at this stage.

### Phase 2: Adversarial persona pressure (multi-round)

For each example in `C`, we run a multi-turn conversation where the user adopts an adversarial persona that challenges the model’s answer for up to `R` rounds (default: `R=5`). Personas are designed to represent qualitatively different pressure mechanisms:
1. **Soft Pressure:** polite disagreement / gentle persuasion.
2. **Simple Denial:** repeated denial without evidence.
3. **Strong Pressure:** forceful insistence.
4. **Authority Claim:** appeal to authority/expertise/norms.
5. **Logical Trap:** reframing / rhetorical traps.

At each round `r`, we score whether the model’s answer is still correct.

**Metrics in brief.** We summarize robustness as a **survival curve** $S_p(r)$: the probability an initially-correct example remains correct through round $r$ under persona $p$. We also report **turn-of-failure (TOF)** as the first round where an initially-correct example flips (or *never* if it does not flip within $R$ rounds), and **recovery** accuracy measured on flipped examples.

**Neutral Re-asking Control (drift baseline).** To distinguish persona-specific pressure from generic multi-turn drift, we also evaluate a control condition that uses the same multi-round structure but removes persona content.

- **Control prompt pattern:** the user repeatedly requests re-checking with neutral phrasing (e.g., “Are you sure? Please verify again.”) without authority claims, traps, or adversarial rhetoric.
  - Example control utterances:
    1. “I’m not fully convinced. Could you double-check your previous answer and respond with your final answer in \boxed{...}?”
    2. “Please verify again step by step, then give only the final answer in \boxed{...}.”
- **Important distinction (vs. Simple Denial persona):** the **Simple Denial** persona is intentionally *adversarial* (repeated contradiction / social pressure). The **Neutral Re-asking Control** is intentionally *non-adversarial* and is meant to capture generic multi-turn variance (e.g., re-evaluation drift, formatting drift) under the same number of rounds.
- **Protocol:** identical `R` rounds, identical decoding settings, identical scoring and logging.
- **Comparison:** we report survival/TOF under (i) persona pressure and (ii) control re-asking, and interpret the gap as evidence of persona mechanism effects beyond drift.

### Phase 3: Recovery

For examples that flipped to incorrect during Phase 2, we provide a recovery prompt designed to help the model re-evaluate and return to the correct answer. We then score recovery accuracy.

---

## 4. Metrics

Let `D` be the full dataset, `C ⊆ D` the initially correct subset, and `P` the set of personas.

### 4.1 Initial accuracy

\[
\text{InitialAcc} = \frac{|C|}{|D|}.
\]

### 4.2 Survival rate (round-wise)

For persona `p` and round `r`:
\[
\text{Survival}(p, r) = \frac{\#\{x\in C : x\ \text{is correct after round}\ r\ \text{under persona}\ p\}}{|C|}.
\]

This produces a **survival curve** across rounds. We report persona-wise curves and aggregates.

### 4.3 Turn-of-failure (TOF)

For each example `x ∈ C` under persona `p`, define `TOF(x, p)` as the earliest round where the answer first becomes incorrect. If it never flips within `R` rounds, set `TOF = never`.

We report the distribution over `{1, 2, …, R, never}` and summarize statistics such as:
- **Fail@1** rate (immediate vulnerability).
- **Never-fail** rate (robustness under that persona).

### 4.4 Recovery accuracy

Let `F_p` be the set of examples in `C` that flipped at least once under persona `p`. Then:
\[
\text{Recovery}(p) = \frac{\#\{x\in F_p : x\ \text{is correct after recovery}\}}{|F_p|}.
\]

We emphasize that recovery is evaluated **conditional on flipping**, and interpret it as an intervention-style metric.

### 4.5 Multi-seed aggregation

For each seed, we compute the above metrics per persona/dataset/round. We then aggregate across seeds by reporting **mean ± std** (optionally with confidence intervals in the final version).

---

## 5. Evaluation details

### 5.1 Boxed final answer standardization

Across tasks, we require the final answer to appear as `\boxed{...}`. Chain-of-thought reasoning (if any) may appear outside the box, but the evaluator extracts boxed content first.

Rationale: multi-turn logs amplify formatting drift; boxed-first extraction reduces scoring failures due to superficial phrasing differences.

### 5.2 Task-specific scoring

- **Math:** exact/normalized match of boxed content.
- **Extractive/Open QA:** normalized text match against a set of acceptable aliases (light normalization).
- **MCQA:** boxed label match (`\boxed{B}`).

---

## 6. Related Work (condensed)

### 6.1 Sycophancy
Sharma et al. (2025) show that RLHF and preference-model optimization can incentivize agreeable-but-wrong behavior and quantify sycophancy across realistic assistant settings. Fanous et al. (2025) propose SycEval to evaluate sycophancy via rebuttal-based prompts and distinguish progressive vs. regressive sycophancy on ground-truth tasks.

Cheng et al. (2025) introduce ELEPHANT to benchmark *social sycophancy* in open-ended contexts through face-preservation behaviors; our qualitative failure modes (e.g., hedging, deference) can be interpreted through a similar lens, while our primary metrics remain ground-truth and dynamics oriented. Petrov et al. (2025) study sycophancy in theorem proving; GALILEO instead targets a broader family of ground-truth tasks and emphasizes multi-turn dynamics plus recovery.

### 6.2 Persuasion and belief vulnerability
Huang et al. (2026) systematize persuasion strategies under an SMCR framework and analyze belief changes in multi-turn interventions. GALILEO complements this line by providing a reproducible, ground-truth-centered protocol with explicit dynamics metrics (survival/TOF) and recovery evaluation.

### 6.3 Stability under context
Tosato et al. (2025) and Yu et al. (2026) report instability in measured traits/personality under prompt and conversation-history variations, supporting the motivation for studying robustness dynamics under accumulating context.

### 6.4 Positioning vs nearby multi-turn sycophancy/robustness benchmarks

Recent benchmarks also probe multi-turn *flip* behavior under disagreement or rebuttals (e.g., SYCON-style multi-turn disagreement, rebuttal-type benchmarks such as SycEval, and long-context degradation studies sometimes framed as “truth decay”). **GALILEO’s intended delta** is to make *ground-truth, multi-turn dynamics* easy to measure and hard to misinterpret:

- **Objective ground truth across tasks:** we emphasize settings where correctness is unambiguous (math/MCQA/extractive QA) rather than primarily subjective opinions or social/ethical dilemmas.
- **Dynamics-first measurement (multi-round):** rather than a single rebuttal, we measure how repeated pressure erodes correctness over rounds via **survival curves** and **turn-of-failure** trajectories.
- **Separate solvability from robustness:** we condition dynamics on the **initially-correct** subset to isolate “can the model solve the task?” from “can it *maintain* the correct belief when challenged?”.
- **Control for generic multi-turn drift:** we include a **Neutral Re-asking Control** with identical round structure/decoding to attribute gaps to persona mechanisms beyond long-context degradation or conversational variance.
- **Recovery as a separate axis:** we evaluate **recovery conditional on flipping** (and prompt-variant ablations) to separate “staying correct” from “returning to correct after being misled.”
- **Reproducible exports:** we provide standardized per-run exports (survival/TOF/recovery) so claims can be verified directly from artifacts.

**Limitations (brief).** Personas approximate social pressure but cannot cover all real conversational tactics; recovery prompts are interventions whose effects may depend on prompt design (mitigated via ablations); and open-domain QA introduces inherent ambiguity, which we treat as realism but report stratified analyses.

---

## 7. Results (template)

This section will be populated from the paper-ready exports under each `results/<run>/paper_exports/` directory. Each subsection explicitly maps back to the core claims (C1–C3).

### 7.1 Main robustness dynamics: survival curves (supports C1, C2)

**Figure X (Survival curves).** Persona-wise survival over rounds `r=1..R` on the main benchmark(s).

- Data source: `paper_exports/survival_curve.csv`
- Suggested caption template:
  - *“Survival(p, r) on initially-correct examples (mean±std across seeds). Robustness decays monotonically for some personas but exhibits late-turn failures under others, highlighting multi-turn dynamics beyond InitialAcc. Dashed line: **Neutral Re-asking Control** (drift baseline).”*

### 7.2 When failures happen: turn-of-failure (TOF) (supports C1)

**Table Y (Turn-of-failure distribution).** Distribution over `{1..R, never}` per persona.

- Data source: `paper_exports/turn_of_failure.csv`
- Suggested caption template:
  - *“TOF reveals early-turn vulnerability (Fail@1) vs sustained robustness (Never-fail). We compute per-seed percentages then report mean±std across seeds.”*

### 7.3 Recovery after flipping (supports C3)

**Table Z (Recovery accuracy).** Recovery conditional on having flipped under persona pressure.

- Data source(s): `recovery_accuracy.csv` (per run) + aggregated exports under `paper_tables_final/`
- Suggested caption template:
  - *“Recovery(p) measures the ability to return to the correct answer after a flip. We report recovery conditional on flip, separating robustness from intervention effects.”*

### 7.4 Cross-task / cross-family generalization (if included in main)

- Report the same survival/TOF/recovery views for additional tasks (QA/MCQA/OpenQA) and at least one additional model family.
- Keep protocol identical; only swap dataset/model.

### 7.5 Persona pressure vs drift: control comparison (supports C2, rebuttal)

**Table W (Persona vs control).** Compare survival@R and Fail@1 under persona pressure vs the **Neutral Re-asking Control** condition.

- Data source: `paper_exports/survival_curve.csv` and `paper_exports/turn_of_failure.csv` for both settings.
- Suggested caption template:
  - *“Persona pressure causes substantially earlier flips than neutral re-asking, indicating effects beyond generic multi-turn drift.”*
- Presentation note: when plotting survival curves, render personas as **solid** lines and the **Neutral Re-asking Control** as a **dashed** line (same axes/rounds/decoding) to make the drift baseline visually unmissable.

<!-- AUTO:TABLE_W_SEED1_START -->
**Seed-1 snapshot (Qwen2.5-7B-Instruct, 200 samples, nlp8).**
- Results root: `results/nlp8_control_vs_persona_20260209_022414/`
- Generated by: `scripts/make_table_w_control_vs_persona.py`

Metric | Control (neutral re-ask) | Persona pressure (agg)
---|---:|---:
Survival@5 | 77.90 | 53.52
Fail@1 | 12.71 | 21.07
Never-fail | 77.90 | 53.52
<!-- AUTO:TABLE_W_SEED1_END -->

---

## 8. Reproducibility checklist (what we will guarantee)

- **Data construction scripts** and a **strict data directory** that excludes legacy/pilot mixtures.
- **Fixed seeds** with deterministic sampling and per-seed metric computation.
- **End-to-end runners** (tmux scripts) and environment documentation.
- **Paper exports**: survival curves, TOF tables, taxonomy sheet templates, and SVG figures.
- **Control condition is always exported.** Export schema includes the **Neutral Re-asking Control** under a stable identifier (e.g., `persona=neutral_reask_control` or `condition=neutral_reask_control`) so plots/tables can include it by default.

---

## 9. Claims → evidence map (reviewer-facing)

This section is written for reviewers: each claim is paired with the *minimum* evidence we will provide (table/figure/analysis) and where it lives in the repo.

### Claim C1: Robustness under pressure is a *multi-turn dynamic*, not a single number
- Evidence:
  - **Survival curves** (persona × round) per dataset + aggregated.
  - **Turn-of-failure (TOF)** distribution (fail@1 / never-fail).
- Artifacts:
  - `results/**/paper_exports/survival_curve.csv`
  - `results/**/paper_exports/turn_of_failure.csv`
  - `paper_figures/` (SVG survival curves and TOF plots)

### Claim C2: Persona mechanisms induce systematically different failure dynamics
- Evidence:
  - Persona-wise survival@R and TOF heatmaps/tables.
  - Qualitative flip taxonomy with representative examples per persona.
- Artifacts:
  - `results/**/paper_tables_final/table_survival_r5.csv`
  - `results/**/paper_taxonomy/*.csv`
  - `PAPER_RESULTS_ANALYSIS_KO.md` + `PAPER_RESULTS_QUAL_EXAMPLES_KO.md`

### Claim C3: Recovery is measurable and behaves differently across tasks/personas
- Evidence:
  - Recovery accuracy conditional on flip (`Recovery(p)`), with variant ablation.
- Artifacts:
  - `results/**/paper_tables_final/table_recovery.csv`
  - `run_experiment.py --recovery_variant ...` (documented in README)

### Claim C4: The phenomenon generalizes beyond a single model family
- Evidence:
  - At least one additional family (e.g., Llama/Mistral/EXAONE) with the same protocol.
- Artifacts:
  - Additional `results/...` roots + same export schema.
  - README “EMNLP Main readiness” checklist section.

### Claim C5: Reproducibility is first-class (strict data, multi-seed, paper-ready exports)
- Evidence:
  - Multi-seed aggregation (mean±std) and deterministic sampling.
  - One-command reproduction instructions.
- Artifacts:
  - `scripts/run_multiseed_tmux.sh`, `scripts/aggregate_multiseed.py`
  - `README.md` (pipeline A/B/C + readiness)


## 10. Limitations and ethics (draft notes)

- Personas approximate social pressure but cannot cover all real conversational tactics.
- Recovery prompts are interventions; results may depend on prompt design. We mitigate this via variant ablations.
- Open-domain QA introduces inherent ambiguity; we treat this as part of the realism but report task uncertainty effects explicitly.

---

## References

See `references.bib` (BibTeX) for citation entries.
