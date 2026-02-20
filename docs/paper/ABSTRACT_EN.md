# GALILEO (EMNLP main) — English abstract draft

Large language models (LLMs) sometimes abandon previously correct answers when users apply conversational pressure (e.g., repeated denial, appeals to authority, persuasive reframing). Standard ground-truth benchmarks largely report single-turn accuracy, obscuring *interaction dynamics*: **when** the first failure occurs, **how** robustness decays across turns, and **whether** a model can return to the correct answer after being misled.

We introduce **GALILEO**, a reproducible protocol for measuring **multi-turn robustness conditional on initial correctness** on ground-truth tasks (math, extractive QA, multiple-choice QA, open-domain QA). We filter to examples answered correctly at round 0, then apply five pressure personas for up to **five rounds**. Each persona arm is paired with a matched **Neutral Re-asking Control (NRC)**: the same multi-round structure and decoding, but with neutral re-check prompts that introduce **no new task-relevant evidence**. Persona–control comparisons are computed on the **same initially-correct subset**, isolating pressure effects from generic multi-turn drift.

GALILEO reports three complementary outcomes: (i) **survival curves**, where **Survival@r** is the cumulative fraction that stays correct at **every** turn through round *r* (i.e., \(\Pr(\forall t\in\{1,\dots,r\}:\,\text{correct at }t\mid\text{correct at }0)\), not “accuracy at round *r* only”); (ii) the **turn-of-failure (TOF)** distribution, with **Fail@1** capturing early-turn vulnerability (first incorrect turn; right-censored if no failure occurs within the horizon); and (iii) **recovery@flip**—accuracy on **one final neutral recovery turn appended after round 5**, conditional on having flipped at least once during rounds 1–5.

Across multi-seed experiments on several open-weight model families, persona pressure consistently reduces survival relative to the NRC and can induce substantial early-turn vulnerability. In our main aggregated setting (seed1–4; persona-weighted average across personas and tasks), persona pressure reduces **Survival@5** by **22.8 percentage points** and increases **Fail@1** by **6.9 points** relative to the NRC. Recovery@flip varies by task and persona, indicating that *staying correct* and *returning to truth after a flip* are distinct, measurable behaviors.

## (Optional) one-sentence positioning

GALILEO turns “LLMs change their minds under pressure” into a measurable, ground-truth, multi-turn robustness + recovery evaluation with matched neutral controls.
