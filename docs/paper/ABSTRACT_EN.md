# GALILEO (EMNLP main) — English abstract draft

Large language models (LLMs) can be pushed off correct answers in interactive settings: under repeated denial, persuasive reframing, or asserted authority, a model may retract a previously correct response even on tasks with a ground-truth label. Prior work documents sycophancy and persuasion effects, but we lack a single, reproducible protocol that (i) localizes *when* failures first occur over multiple turns, (ii) characterizes multi-turn *survival dynamics* beyond single-turn accuracy, and (iii) measures whether models can *recover* after flipping.

We introduce **GALILEO**, a benchmark and evaluation pipeline for ground-truth tasks (math, extractive QA, multiple-choice QA, and open-domain QA) that quantifies **multi-turn robustness conditional on initial correctness** under adversarial social pressure. GALILEO (1) evaluates initial correctness, (2) applies five adversarial personas—Soft Pressure, Simple Denial, Strong Pressure, Authority Claim, and Logical Trap—for up to five rounds and tracks survival curves and the **turn-of-failure (TOF)** distribution (including **Fail@1**), and (3) prompts a recovery intervention after a flip to measure **Recovery@flip**. Each persona arm is paired with a **Neutral Re-asking Control (NRC)** that matches dialogue length and decoding while introducing **no new task-relevant evidence**; persona–control comparisons are computed on the **same initially-correct subset** to isolate pressure effects from generic multi-turn drift. To enable stable automated scoring across tasks, we standardize final answers using a `\boxed{...}` format.

Across main experiments (Qwen2.5-7B-Instruct, seeds 1–4) and additional model families (e.g., Mistral-7B and Llama-3.1-8B, seeds 1–2), persona pressure consistently degrades survival and increases early-turn failures relative to the NRC drift baseline under an identical protocol. Recovery effects vary by persona and task, suggesting robustness (staying correct) and recovery (return-to-truth after flipping) are distinct axes. A decoding-sensitivity check further shows the persona–control gap persists across temperatures.

GALILEO provides an auditable, reproducible way to measure belief-consistency vulnerabilities and recovery behavior in multi-turn, ground-truth interaction settings.

## (Optional) one-sentence positioning

GALILEO turns “LLMs change their minds under pressure” into a measurable, multi-turn robustness + recovery evaluation with matched neutral controls and tracked artifacts.
