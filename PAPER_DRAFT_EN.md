# GALILEO Paper Draft (EN) — Working skeleton

> Status: early English scaffold for submission writing. Source of truth for experiments/results remains in the repo; this file is for *paper-quality phrasing*.

---

## Abstract

Large language models (LLMs) can exhibit *belief-consistency failures* under social and rhetorical pressure—e.g., repeated denial, authoritative claims, or persuasive reframing—sometimes retracting previously correct answers even on tasks with clear ground truth. Prior work studies sycophancy and persuasion-related vulnerabilities (e.g., Sharma et al., 2025; Fanous et al., 2025; Huang et al., 2026), but a unified, reproducible protocol that measures **when** failures first occur (**turn-of-failure**), **how** robustness evolves over interaction rounds (**survival curves**), and **whether** models can recover after flipping (**recovery**) on ground-truth tasks remains limited.

We present **GALILEO**, a benchmark and pipeline for measuring multi-turn robustness under adversarial *persona-based* pressure. GALILEO (i) evaluates initial correctness, (ii) applies five adversarial personas for up to five rounds and measures round-wise survival, and (iii) evaluates recovery on samples that flipped. To ensure stable automatic scoring across tasks and multi-turn logs, we standardize the final answer format to `\boxed{...}` and perform boxed-first extraction in the evaluator. Initial snapshots indicate persona-dependent robustness dynamics, sharp degradation on uncertain open-domain QA, and task-dependent recovery patterns.

---

## 1. Introduction (draft)

**Problem.** In high-stakes applications (education, healthcare, legal assistance), it is not enough for an LLM to be correct once; it must remain correct under interaction pressure. Multi-turn conversations introduce repeated challenges and rhetorical leverage, creating a failure mode where the model *abandons* a correct answer.

**Gap.** Single-turn accuracy does not capture (i) *when* a model first fails under pressure, (ii) how failure probability compounds over rounds, or (iii) whether the model can recover after being misled.

**Contributions.**
1. **Ground-truth multi-turn dynamics.** We operationalize robustness as survival curves and turn-of-failure, and measure recovery in the same protocol.
2. **Unified multi-task pipeline.** We cover math, extractive QA, MCQA, and open-domain QA with a single runner/logging/evaluation interface.
3. **Reproducibility + exports.** We provide strict data splits, multi-seed aggregation, and paper-ready exports (tables/figures).

---

## 2. Related Work (draft)

### 2.1 Sycophancy
- **Sycophancy in RLHF assistants.** Sharma et al. (2025) analyze sycophancy across RLHF-trained assistants and show that human preference data and preference-model optimization can incentivize agreeable-but-wrong responses.
- **SycEval.** Fanous et al. (2025) propose rebuttal-based evaluation and distinguish progressive vs. regressive sycophancy on ground-truth settings.
- **Social sycophancy.** Cheng et al. (2025) introduce ELEPHANT to measure face-preservation behavior in open-ended contexts; our qualitative failure modes (hedging/deference) can be interpreted through this lens.
- **Formal reasoning.** Petrov et al. (2025) study sycophancy in theorem proving; GALILEO targets a broader set of ground-truth tasks and emphasizes dynamics + recovery.

### 2.2 Persuasion and belief vulnerability
Huang et al. (2026) systematize persuasion strategies under an SMCR framework and analyze when belief changes occur; our turn-of-failure and survival curves provide a complementary, reproducible measurement on ground-truth tasks.

### 2.3 Stability/instability under context
Tosato et al. (2025) and Yu et al. (2026) report instability in measured traits/personality across prompts and conversation histories, supporting the motivation to study robustness dynamics under accumulating context.

---

## References

See `references.bib` (BibTeX) for citation entries.
