# GALILEO (EMNLP main) — English abstract draft

Large language models (LLMs) can retract previously correct answers under conversational pressure (e.g., repeated denial, appeals to authority, persuasive reframing). Standard ground-truth benchmarks largely report single-turn accuracy, obscuring *interaction dynamics*: **when** a model first flips from correct to incorrect, **how** robustness decays over rounds, and **whether** it can return to the correct answer after being misled.

We introduce **GALILEO**, a reproducible protocol for measuring **multi-turn robustness conditional on initial correctness** on ground-truth tasks (math, extractive QA, multiple-choice QA, open-domain QA). We first filter to examples answered correctly at round 0, then apply five pressure personas for up to **five rounds**. Each persona arm is paired with a matched **Neutral Re-asking Control (NRC)**: a neutral re-check prompt repeated under identical rounds/decoding that introduces **no new task-relevant evidence**. To attribute effects to pressure (rather than generic multi-turn drift), **persona–control comparisons are computed on the same initially-correct subset**.

GALILEO reports three complementary outcomes: (i) **survival curves**, where **Survival@r** is the cumulative fraction that stays correct at **every** turn through round *r* (i.e., \(\Pr(\forall t\in\{1,\dots,r\}:\,\text{correct at }t\mid\text{correct at }0)\), not “accuracy at round *r* only”), (ii) the **turn-of-failure (TOF) distribution** with **Fail@1** capturing early-turn vulnerability (first incorrect turn; right-censored if no failure within the horizon), and (iii) **recovery@flip**—accuracy on a final neutral recovery prompt, **conditional on having flipped at least once during rounds 1–5** (the recovery turn is excluded from the flip/TOF definition).

**Timeline note:** rounds 1–5 are persona pressure (or NRC re-asks); **after** round 5 we append **one additional neutral recovery turn** (identical across arms) solely to measure recovery@flip.

Across multi-seed experiments on several open-weight model families, persona pressure consistently reduces survival relative to the NRC and can induce substantial early-turn vulnerability. In our main aggregated setting (seed1–4; persona-weighted average across personas and tasks), persona pressure reduces **Survival@5** by **22.8 percentage points** and increases **Fail@1** by **6.9 points** relative to the NRC. Recovery@flip varies by task and persona, indicating that *staying correct* and *returning to truth after a flip* are distinct, measurable behaviors.

## (Optional) one-sentence positioning

GALILEO turns “LLMs change their minds under pressure” into a measurable, ground-truth, multi-turn robustness + recovery evaluation with matched neutral controls.
