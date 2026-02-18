# OS-Harm: A Benchmark for Measuring Safety of Computer Use Agents

- Year: 2025
- Venue: NeurIPS 2025 Datasets & Benchmarks Track (Spotlight)
- Authors: Thomas Kuntz, Agatha Duzan, Hao Zhao, Francesco Croce, Zico Kolter, Nicolas Flammarion, Maksym Andriushchenko
- URL: https://arxiv.org/abs/2506.14866
- BibTeX key (if we add it): osharm2025kuntz
- Tags: agents, computer-use, safety, benchmark, prompt-injection, misuse, evaluation, llm-judge

## One-sentence takeaway

OS-Harm is a realistic OSWorld-based benchmark (150 tasks) that measures *agent* safety (misuse, prompt injection, and misbehavior) for GUI computer-use agents, using an automated semantic judge with decent agreement to human labels.

## What problem does it solve?

- Safety for “computer use agents” (GUI agents operating across apps) is under-evaluated compared to chatbot safety.
- Existing safety benchmarks often emulate tools/side effects, whereas computer-use agents can take real multi-step actions in an OS environment.
- Need a benchmark that separates key harm sources:
  - deliberate user misuse,
  - third-party prompt injection,
  - model misbehavior / costly mistakes during benign tasks.

## What is the core method / protocol?

- Benchmark built on **OSWorld** (Ubuntu desktop VM tasks) with **150 tasks** total.
- Tasks cover **3 categories** (50 each):
  1) **Deliberate user misuse** (user directly requests harmful goals).
  2) **Prompt injection attacks** via untrusted third-party content (emails/webpages/notifications).
  3) **Model misbehavior** on benign tasks that can still cause harm (e.g., privacy leaks, costly mistakes).
- Tasks span multiple **apps** (paper mentions e.g., Thunderbird, VS Code, Terminal, LibreOffice Impress) and safety violation types (harassment, copyright infringement, disinformation, data exfiltration, etc.).
- Evaluation uses an **automated LLM judge** that scores both:
  - task completion / accuracy
  - safety (whether unsafe actions were taken / policy violations)
  using execution traces (reasoning steps + GUI screenshots + summarized a11y trees).

## What are the key metrics?

- Dual evaluation:
  - **Accuracy** (task success)
  - **Safety** (harmful/unsafe behavior)
- Judge quality reported as agreement with human annotations:
  - ~**0.76 F1** (accuracy)
  - ~**0.79 F1** (safety)
- Paper also reports operational/benchmark stats (from the HTML):
  - total tasks: 150
  - distinct OS applications used: 11
  - distinct files used: 53
  - average config entries per task: 3.26
  - example run cost/time reported for an o4-mini agent (as of May 2025)

## What are the main results?

- Across several frontier models (paper mentions o4-mini, Claude 3.7 Sonnet, Gemini 2.5 Pro):
  - Models **often comply** with deliberate misuse requests (misuse vulnerability).
  - Models are **vulnerable to static prompt injections** (paper gives an example rate: ~20% for o4-mini in basic prompt-injection cases).
  - **Model misbehavior** occurs less frequently but can still be costly/unsafe.
- Automated judging appears viable for this setting with the reported F1 agreement.

## How is this similar to GALILEO?

- Shared concern: evaluating *agentic* behavior under adversarial or unsafe pressures, not just single-turn text.
- Uses **protocolized tasks + metrics** to quantify safety failures and robustness.
- Includes prompt-injection as a first-class phenomenon (a key mechanism for “drift” of intent/behavior in agent loops).

## How is this different from GALILEO?

- OS-Harm focuses on **GUI computer-use** safety (OSWorld) and *harm categories*; GALILEO appears centered on different failure modes/metrics (e.g., behavioral instability / drift / evaluation protocols).
- Evaluation relies on an **LLM judge over traces**; GALILEO may prefer more controlled measurement designs and/or different outcome metrics.
- “Safety” here is primarily policy-violation / harmful-action oriented (misuse / injection / misbehavior), rather than belief revision vs drift or recovery dynamics.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO emphasizes **control conditions** (drift vs evidence-based revision) and **time-to-failure / recovery** metrics, it may yield cleaner causal statements than a broad harm taxonomy benchmark.
- If GALILEO uses more systematic robustness metrics, it could complement OS-Harm’s categorical safety framing.

## Where GALILEO is weaker / needs to improve

- OS-Harm highlights the need for **realistic, end-to-end agent environments** where unsafe outcomes are actionable; if GALILEO is more abstract/textual, it may miss GUI-specific attack surfaces.
- LLM-judge-based scoring (with validated agreement) is a pragmatic approach; if GALILEO lacks scalable evaluation, OS-Harm is a useful reference.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding (or explicitly contrasting with) a **prompt-injection-in-the-loop** condition, including untrusted third-party content that competes with the system/task objective.
- [ ] If using automatic evaluation, cite OS-Harm as evidence that **semantic judges over traces** can reach reasonable agreement with humans (reporting F1s).
- [ ] Consider adopting a similar **harm taxonomy** framing (misuse vs injection vs misbehavior) as an organizing lens for failure analysis.

## Quotes / details to potentially cite

- “OS-Harm is built on top of the OSWorld environment … and aims to test models across three categories of harm: deliberate user misuse, prompt injection attacks, and model misbehavior.”
- “We create 150 tasks … require the agent to interact with a variety of OS applications … (email client, code editor, browser, etc.).”
- “We propose an automated judge to evaluate both accuracy and safety … high agreement with human annotations (0.76 and 0.79 F1 score).”
