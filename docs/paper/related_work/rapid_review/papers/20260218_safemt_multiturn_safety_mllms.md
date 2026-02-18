# SafeMT: Multi-turn Safety for Multimodal Language Models

- Year: 2025
- Venue: arXiv
- Authors: Han Zhu; Juntao Dai; Jiaming Ji; Haoran Li; Chengkun Cai; Pengcheng Wen; Chi-Min Chan; Boyuan Chen; Yaodong Yang; Sirui Han; Yike Guo
- URL: https://arxiv.org/abs/2510.12133
- BibTeX key (if we add it): safemt2025zhu
- Tags: multimodal, multi-turn, safety, jailbreak, benchmark

## One-sentence takeaway

SafeMT is a 10k-sample benchmark for evaluating how multimodal LLM safety degrades across *multi-turn* harmful dialogues (with images), plus a proposed “Safety Index” and a dialogue safety moderator that reduces multi-turn attack success.

## What problem does it solve?

- Existing safety benchmarks for MLLMs focus heavily on single-turn prompts and under-represent the more realistic setting where harmful intent is gradually revealed across turns (and may be coupled with images).
- Lack of a standardized way to measure *overall* conversational safety across varying dialogue lengths and jailbreak strategies.

## What is the core method / protocol?

- Construct a benchmark (SafeMT) of multi-turn dialogues of varying lengths, generated from harmful queries + images.
- Cover multiple “scenarios” (17) and multiple jailbreak methods (4).
- Evaluate multiple MLLMs (17 models reported) on the benchmark.
- Propose:
  - Safety Index (SI): a scalar score intended to summarize general safety during conversation (details not in abstract).
  - A dialogue safety moderator that (i) detects malicious intent hidden across turns and (ii) supplies the MLLM with relevant safety policies.

## What are the key metrics?

- Attack success rate (ASR) in multi-turn harmful dialogues (implied).
- Safety Index (SI) (their proposed aggregate metric).
- Sensitivity of ASR/SI to number of dialogue turns.

## What are the main results?

- Across 17 evaluated models, successful attack risk increases as the number of turns increases.
- The proposed dialogue safety moderator reduces multi-turn ASR more effectively than existing “guard models” on several open-source MLLMs.

## How is this similar to GALILEO?

- Shares a *multi-turn protocol* emphasis: failures often emerge only after several turns, so evaluation should measure degradation over time/turns rather than only single-turn behavior.
- Benchmark framing: multiple scenarios + structured attack methods resembles the need for structured stress-testing protocols.

## How is this different from GALILEO?

- Focuses on safety/jailbreak and harmful intent concealment, not belief/stance stability under social pressure (GALILEO’s core).
- Multimodal (images + text) is central here; GALILEO may be primarily text/social interaction (depending on the exact setup).
- Introduces a moderator/policy-injection approach rather than modeling or measuring opinion dynamics/stance drift per se.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO isolates *belief change mechanisms* (evidence-driven revision vs social pressure) with explicit controls, it may provide cleaner causal attribution than a broad safety ASR benchmark.
- GALILEO may offer more fine-grained trajectory metrics for “drift” (e.g., turn-of-failure / time-to-event style measures) if implemented.

## Where GALILEO is weaker / needs to improve

- If GALILEO lacks multimodal attack surfaces and multi-turn “concealed intent” scenarios, SafeMT highlights an important class of multi-turn failure modes.
- If GALILEO reports only end-state metrics, SafeMT underscores that risk grows with turns and should be measured as a function of dialogue length.

## Action items for GALILEO (experiments / method / writing)

- [ ] In related work, cite SafeMT as evidence that *multi-turn interaction length amplifies risk*; motivate reporting metrics vs number of turns (or survival/time-to-failure curves).
- [ ] Consider adding a “concealed intent” / gradual-reveal variant to GALILEO prompts (even if not safety-focused), to test whether instability emerges only after long-context social maneuvering.
- [ ] If applicable, compare against a simple “moderator” baseline (policy reminder / intent detector) to see whether instability can be reduced by intervention.

## Quotes / details to potentially cite

- “Multi-turn dialogues … pose a greater risk than single prompts; however, existing benchmarks do not adequately consider this situation.” (abstract)
- “the risk of successful attacks … increases as the number of turns in harmful dialogues rises.” (abstract)
- SafeMT: “10,000 samples … 17 different scenarios and four jailbreak methods.” (abstract)
