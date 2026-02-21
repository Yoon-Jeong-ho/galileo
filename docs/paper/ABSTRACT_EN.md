# GALILEO (EMNLP main) — English abstract draft

Large language models (LLMs) can abandon previously correct answers when users apply conversational pressure (e.g., repeated denial, appeals to authority, persuasive reframing). Yet standard ground-truth benchmarks largely report single-turn accuracy, obscuring interaction dynamics: when the first failure occurs, how robustness decays across turns, and whether a model can return to the correct answer after being misled.

We introduce **GALILEO**, a reproducible protocol for measuring **multi-turn robustness conditional on initial correctness** on ground-truth tasks (math, extractive QA, multiple-choice QA, open-domain QA). For each task, we first filter to examples answered correctly at round 0, then apply five pressure personas for up to five rounds. Each persona arm is paired with a matched **Neutral Re-asking Control (NRC)** that preserves the same multi-round structure and decoding but uses neutral re-check prompts that introduce no new task-relevant evidence. By comparing persona vs. control on the same initially-correct subset, GALILEO isolates pressure effects from generic multi-turn drift.

GALILEO reports three complementary outcomes: (i) **Survival@r**, the fraction of initially-correct examples that remain correct at every turn through round *r*; (ii) the **turn-of-failure (TOF)** distribution, including **Fail@1** (first incorrect turn; right-censored if no failure occurs within the horizon); and (iii) **Recovery@flip**, accuracy on one final neutral recovery turn appended after round 5, conditional on having flipped at least once during rounds 1–5 (excluded from Survival/TOF).

Across multi-seed experiments on several open-weight model families, persona pressure consistently reduces survival relative to the NRC and increases early-turn vulnerability (e.g., −22.8 pp Survival@5 and +6.9 pp Fail@1 in our main aggregated setting). Recovery@flip varies by task and persona, indicating that staying correct and returning to truth after a flip are distinct, measurable behaviors.

## (Optional) one-sentence positioning

GALILEO turns “LLMs change their minds under pressure” into a measurable, ground-truth, multi-turn robustness-and-recovery evaluation with matched neutral controls.
