# SafeDialBench: A Fine-Grained Safety Evaluation Benchmark for Large Language Models in Multi-Turn Dialogues with Diverse Jailbreak Attacks

- Year: 2025
- Venue: arXiv
- Authors: Hongye Cao, Sijia Jing, Yanming Wang, Ziyue Peng, Zhixin Bai, Zhe Cao, Meng Fang, Fan Feng, Boyan Wang, Jiaheng Liu, Tianpei Yang, Jing Huo, Yang Gao, Fanyu Meng, Xi Yang, Chao Deng, Junlan Feng
- URL: https://arxiv.org/abs/2502.11090
- BibTeX key (if we add it): safedialbench_cao_2025
- Tags: multi-turn, safety, jailbreak, benchmark, evaluation, taxonomy

## One-sentence takeaway

SafeDialBench is a fine-grained multi-turn *safety* benchmark with a hierarchical taxonomy and diverse jailbreak strategies, aimed at measuring not just whether models fail, but whether they can **detect**, **handle**, and **stay consistent** when exposed to unsafe content over dialogue turns.

## What problem does it solve?

- Existing safety benchmarks skew toward **single-turn** prompts or **one jailbreak style**, and do not measure *in detail* how well an LLM can (i) identify unsafe information, (ii) respond appropriately, and (iii) maintain safety behavior consistently across turns.

## What is the core method / protocol?

- Proposes a **two-tier hierarchical safety taxonomy** spanning **6 safety dimensions**.
- Builds **4k+ multi-turn dialogues** (Chinese + English) across **22 scenarios**.
- Applies **7 jailbreak attack strategies** (examples mentioned: *reference attack*, *purpose reverse*) to generate/augment adversarial multi-turn dialogues.
- Evaluation framing emphasizes three capabilities:
  - **Detecting** unsafe information
  - **Handling** unsafe information
  - **Maintaining consistency** under multi-turn jailbreak pressure

## What are the key metrics?

- Paper’s headline metrics are capability-oriented (detect / handle / consistency) rather than a single scalar “jailbreak success rate”.
- (From abstract) the benchmark’s assessment framework measures:
  - detection ability
  - handling ability
  - consistency under attack

## What are the main results?

- Evaluated **17 LLMs**.
- (From abstract) Yi-34B-Chat and GLM4-9B-Chat show stronger safety performance; Llama3.1-8B-Instruct and o3-mini show notable safety vulnerabilities.

## How is this similar to GALILEO?

- Both are **multi-turn robustness** evaluations where performance degrades (or policies are circumvented) under **adversarial follow-ups**.
- The “maintaining consistency across turns” framing is aligned with GALILEO’s focus on *trajectory-level* outcomes (not only first-turn accuracy).

## How is this different from GALILEO?

- Target phenomenon: **safety / jailbreak resilience** rather than belief stability, persuasion-induced drift, or truthfulness under social pressure.
- Outputs are judged primarily by safety taxonomy compliance (detect/handle/consistency), not by “stay with the truth unless new evidence” style controls.
- Includes **bilingual** multi-turn dialogues and scenario coverage that is broader than GALILEO’s current robustness slices.

## Where GALILEO is stronger / cleaner (if true)

- GALILEO’s positioning around **drift vs evidence-driven revision** (and robustness metrics like time-to-failure / recovery) can be conceptually cleaner for *truth/stance stability* than a pure jailbreak benchmark.

## Where GALILEO is weaker / needs to improve

- If we want to claim broader “multi-turn robustness,” we likely need at least a small slice or discussion connecting **jailbreak-style adversarial dialogue** to our operators/pressure taxonomy.
- Fine-grained capability decomposition (detect vs handle vs consistency) is a useful complement to our current reporting.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a paragraph in related work contrasting **multi-turn persuasion/consistency** benchmarks vs **multi-turn safety/jailbreak** benchmarks; cite SafeDialBench as a representative “fine-grained multi-turn safety” dataset.
- [ ] Consider a lightweight “safety-consistency under adversarial dialogue” appendix experiment (even if out-of-scope) to argue generality of our trajectory metrics.
- [ ] Borrow their **capability decomposition** idea: report a separation between (i) detecting misleading/unsafe pressure, (ii) handling it appropriately, (iii) staying consistent across turns.

## Quotes / details to potentially cite

- “We propose a fine-grained benchmark SafeDialBench for evaluating the safety of LLMs across various jailbreak attacks in multi-turn dialogues.”
- “We design a two-tier hierarchical safety taxonomy that considers 6 safety dimensions and generates more than 4000 multi-turn dialogues in both Chinese and English under 22 dialogue scenarios.”
- “Notably, we construct an innovative assessment framework of LLMs, measuring capabilities in detecting, and handling unsafe information and maintaining consistency when facing jailbreak attacks.”
