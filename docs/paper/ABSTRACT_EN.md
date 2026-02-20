# GALILEO (EMNLP main) — English abstract draft

Large language models (LLMs) can retract previously correct answers under conversational pressure (e.g., repeated denial, appeals to authority). Standard benchmarks typically report single-turn accuracy, obscuring *interaction dynamics*: **when** a model first flips, **how** robustness decays over rounds, and **whether** it can return to the correct answer.

We introduce **GALILEO**, a reproducible protocol for **multi-turn robustness conditional on initial correctness** on ground-truth tasks. We first filter to examples answered correctly at round 0, then apply five pressure personas for up to **five rounds**. Each persona arm is paired with a matched **Neutral Re-asking Control (NRC)**: the same multi-round scaffold and decoding, but with a strictly neutral re-check prompt that introduces **no new task-relevant evidence**. Persona–control comparisons are computed on the **same initially-correct subset**, separating pressure-induced flips from generic multi-turn drift.

GALILEO reports three outcomes: (i) **survival curves** (fraction remaining correct through round *r*), (ii) **turn-of-failure (TOF)** with **Fail@1** capturing early-turn vulnerability, and (iii) **recovery@flip**—accuracy on a final neutral recovery prompt, conditional on having flipped at least once.

Across multi-seed experiments on several open-weight model families, persona pressure reduces survival relative to the NRC; recovery@flip varies by task and persona, indicating that *staying correct* and *returning to truth after a flip* are distinct behaviors.

## (Optional) one-sentence positioning

GALILEO turns “LLMs change their minds under pressure” into a measurable, ground-truth, multi-turn robustness + recovery evaluation with matched neutral controls.
