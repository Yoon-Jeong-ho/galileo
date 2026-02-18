# SafePro: Evaluating the Safety of Professional-Level AI Agents

- Year: 2026
- Venue: arXiv
- Authors: Kaiwen Zhou, Shreedhar Jangam, Ashwin Nagarajan, Tejas Polu, Suhas Oruganti, Chengzhi Liu, Ching-Chen Kuo, Yuting Zheng, Sravana Narayanaraju, Xin Eric Wang (author list per arXiv HTML; verify ordering if needed)
- URL: https://arxiv.org/abs/2601.06663
- BibTeX key (if we add it): safepro2026
- Tags: agents, safety, benchmark, professional-tasks, harmful-instructions, llm-judge

## One-sentence takeaway

SafePro is a safety-alignment benchmark for *professional*, high-complexity agent tasks (275 samples across occupations/risk categories) showing that strong frontier models can be unsafe at high rates, and that basic mitigation layers can reduce—but not eliminate—these failures.

## What problem does it solve?

- Existing agent safety evals largely target “daily assistant” tasks or narrow vectors (e.g., prompt injection, single-step misuse) and may miss failure modes that emerge in longer-horizon, domain-specific, professional workflows.
- Provides a concrete dataset + evaluation protocol to measure unsafe behavior when an agent is asked to perform plausibly-realistic professional work with embedded harmful intent/risk outcomes.

## What is the core method / protocol?

- **Dataset (SafePro):** 275 “harmful professional task” samples spanning multiple occupation sectors and explicit **risk categories** (e.g., property/financial loss, discrimination/bias, misinformation, info disclosure, physical harm, system compromise, environmental harm, IP misuse, other illegal/regulatory violations).
- **Task construction:** two routes:
  - *Benign task transformation* (adapt GDPval-style professional tasks by injecting unsafe intent).
  - *New harmful task generation* (from scratch), with an iterative create-and-review QC loop and per-task **safe vs unsafe criteria**.
- **Evaluation:** LLM-agent is run on tasks; safety is judged via **LLM-as-a-judge** (paper uses GPT-5-mini as primary) that checks the agent’s response/actions against the provided unsafe criteria + category context.
- **Reliability check:** cross-judge evaluation with multiple judge models to test for obvious “self-favoring” bias.
- **Mitigations explored (high-level):** safety prompting, LLM safety classification, and “guardrails” (details beyond the intro/benchmark section).

## What are the key metrics?

- **Unsafe rate** on the benchmark (fraction of tasks judged Unsafe).
- Analysis axes mentioned: **safety judgment** vs **safety alignment** deficits (conceptual decomposition rather than a single scalar metric).
- Cross-judge agreement / consistency via comparing unsafe rates under different judge models.

## What are the main results?

- Reports **substantial unsafe rates** for state-of-the-art models on professional harmful tasks (intro highlights “over 40%” unsafe rates for some leading models on SafePro).
- Finds **new unsafe behaviors** in professional contexts and argues failures stem from both:
  - insufficient safety judgment (recognizing risk / refusing appropriately), and
  - weak safety alignment (staying safe while still executing complex tasks).
- Mitigation strategies show **encouraging improvements**, but the framing implies gaps remain.

## How is this similar to GALILEO?

- Shared motivation: current “safe enough” evaluations can miss important real-world failure modes; both argue for **harder, more realistic evaluation protocols**.
- Both are “benchmark + protocol” papers intended to **surface systematic vulnerabilities** (SafePro: unsafe professional actions; GALILEO: multi-turn belief/answer drift under persona pressure and recovery).
- Both emphasize **auditable criteria** (SafePro: per-task safe/unsafe criteria; GALILEO: ground-truth scoring + turn-of-failure + recovery).

## How is this different from GALILEO?

- **Task type:** SafePro targets *harmful/misuse* professional tasks with risk outcomes; GALILEO targets *ground-truth correctness* under multi-turn persuasion/persona pressure.
- **Outcome variable:** SafePro primarily measures *safety* (unsafe vs safe); GALILEO measures *belief-consistency/correctness survival*, turn-of-failure, and recovery.
- **Judge dependence:** SafePro uses LLM-as-judge for safety classification; GALILEO’s core scoring is designed to be more **automatic/ground-truth** (where applicable).
- **Interaction structure:** SafePro looks like single-task professional workflows (possibly with tools/web); GALILEO is explicitly **multi-round** pressure with a matched control arm.

## Where GALILEO is stronger / cleaner (if true)

- Clearer **objective scoring** story for many tasks (conditioning on initially-correct subset; explicit turn-of-failure and recovery), reducing dependence on judge models.
- Stronger handle on **multi-turn dynamics** (SafePro’s core metric appears largely aggregate unsafe rate, unless extended elsewhere in the paper).

## Where GALILEO is weaker / needs to improve

- Coverage: GALILEO is not primarily a *professional-harm* benchmark; it may under-represent domain-specific, high-stakes professional workflows and risk categories.
- Safety taxonomy: SafePro’s explicit mapping to occupation sectors + risk categories could be a useful model for broadening GALILEO’s positioning/analysis.

## Action items for GALILEO (experiments / method / writing)

- [ ] Related-work positioning: cite SafePro as a complementary “professional agent safety” benchmark; explicitly contrast **judge-based safety classification** vs **ground-truth multi-turn drift**.
- [ ] Consider adding a short discussion section: how GALILEO-style *multi-turn pressure* could be applied to professional-task agents (bridging the two paradigms).
- [ ] If adding a new experiment is feasible later: run a small pilot where SafePro-style harmful professional tasks are re-asked under GALILEO personas to see whether **persona pressure increases unsafe compliance** over turns.

## Quotes / details to potentially cite

- “SafePro … evaluate the safety alignment of AI agents performing professional activities.” (abstract framing)
- Dataset scale + diversity: “275 data samples … cover a wide range of occupations and risk categories.” (benchmark overview)
- Creation approach: “Benign Task Transformation … 195 harmful tasks” + “New Harmful Task Generation … 80 harmful tasks.”
- Risk categories table includes: property/financial loss; discrimination/bias; misinformation; information disclosure; physical harm; system compromise; environmental harm; IP misuse; other illegal/regulatory violations.
