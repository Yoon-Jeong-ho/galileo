# FURINA: A Fully Customizable Role-Playing Benchmark via Scalable Multi-Agent Collaboration Pipeline

- Year: 2025
- Venue: arXiv
- Authors: Haotian Wu; Shufan Jiang; Chios Chen; Yiyang Feng; Hehai Lin; Heqing Zou; Yao Shu; Yanran Li; Chengwei Qin
- URL: https://arxiv.org/abs/2510.06800
- BibTeX key (if we add it): furina2025
- Tags: role-play, benchmark, benchmark-builder, multi-agent, llm-judge, evaluation-dimensions, hallucination

## One-sentence takeaway

FURINA proposes a multi-agent “benchmark builder” that can generate customizable role-playing evaluation sets (characters, scenarios, prompts, dimensions), and uses a dimension-selecting LLM judge to create higher-quality, more separable RP benchmarks while highlighting an RP-performance vs hallucination trade-off.

## What problem does it solve?

- Existing role-playing (RP) benchmarks are mostly static: fixed characters, limited scenario coverage, limited prompt formats, and fixed (or noisy) evaluation dimensions.
- Because RP products/characters change quickly and differ across platforms/use cases, static benchmarks become obsolete and are hard to tailor to a specific target character design.
- Evaluating RP well also needs fine-grained, context-appropriate dimensions (not “score everything at once”).

## What is the core method / protocol?

- **FURINA-Builder**: a multi-agent pipeline to synthesize RP benchmark items by simulating multi-party conversations and selecting/labeling evaluation dimensions.
  - Inputs are customizable: test character definition (key/value attributes with **public/private visibility**), character-scene pool, dialogue structure, and evaluation dimensions.
  - Components:
    - **Character–scene pool**: scenario fragments (background, motivations, original dialogue reference, scene characters) curated from bilingual books; user can extend.
    - **Simulation**: a “director” chooses next speaker; test character interacts with scene characters; private attributes can be hidden to create more realistic difficulty.
    - **Two candidate responses** for the test character each turn: from a **source model** (under test / data source) and a strong **base model** (floor/reference).
    - **Judge model** does two things per test-character turn:
      1) selects a single most appropriate evaluation dimension for that utterance;
      2) performs pairwise judgment on a Likert scale, then picks the better candidate to continue the conversation.
  - The paper argues single-dimension selection reduces cross-dimension interference and avoids forcing irrelevant dimensions on a given utterance.

- **FURINA-Bench**: benchmark produced by the builder.
  - Mixes **established characters** and **synthesized characters**.
  - Uses **group-chat / multi-character** conversations.
  - Associates each test utterance with one dimension label.

## What are the key metrics?

- Five evaluation dimensions (used during construction and evaluation):
  - **Context Reliance (CR)**
  - **Factual Recall (FR)**
  - **Reflective Reasoning (RR)**
  - **Conversational Ability (CA)**
  - **Preference Alignment (PA)** (penalizes robotic/repetitive single-turn responses)
- Dimension-selection accuracy (judge picking the “right” dimension) via human-annotated checks.
- Correlation between judge scores and human annotations (Pearson correlation reported for dimensions).
- RP hallucination rate / severity (paper emphasizes a reliability trade-off; details are later in the paper beyond the excerpt captured).

## What are the main results?

- Reported **dimension selection accuracy** (GPT-4.1 judge) around **0.892 average** over 5 dimensions on a 1000-sample check (table shown).
- Judge-vs-human correlations are reasonably strong across dimensions; GPT-4.1 judge aligns better than alternatives (DeepSeek variants) in their reported table.
- Benchmark scale (reported): **20 test characters** (Chinese/English; established/synthesized), ~**1,459** multi-party dialogues, ~**7,181** test utterances; balanced dimensions/languages.
- Model eval headline (as stated in abstract/intro):
  - o3 best overall on English RP tasks; DeepSeek-R1 best on Chinese.
  - Established characters generally easier / higher performance than synthesized; reasoning can widen the gap.
  - Key qualitative claim: **reasoning improves RP performance but increases RP hallucinations**, yielding a Pareto frontier between performance and reliability.

## How is this similar to GALILEO?

- Both are about **evaluation frameworks** that try to be robust and practically useful (not just single static test sets).
- The “dimension selection + targeted judging” idea is adjacent to GALILEO-style concerns about **metric validity**, **separability**, and avoiding conflated/overloaded evaluation prompts.
- Highlights a concrete **performance vs reliability** tension (useful framing for GALILEO’s reliability/robustness discussions).

## How is this different from GALILEO?

- FURINA is specifically focused on **role-playing conversational agents** (characters, scenarios, multi-party chat), rather than general-purpose assistant evaluation.
- It emphasizes **benchmark construction via multi-agent simulation** and uses a strong base model in the loop to keep trajectories “high quality” (which may bias distributions).
- Uses a **single-dimension-per-utterance** labeling/judging protocol; GALILEO may instead evaluate with broader task suites or different decomposition.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO targets broader assistant behavior (tools, factual QA, long-horizon tasks), it can claim wider coverage beyond RP.
- GALILEO can position itself as less dependent on a specific in-the-loop “base model” during data generation (potentially reducing benchmark contamination / anchoring).

## Where GALILEO is weaker / needs to improve

- If GALILEO currently lacks a strong **benchmark builder** story (customizable generation pipeline), FURINA is a compelling reference point.
- The “dimension-selection” mechanism suggests GALILEO should be careful about **multi-dimension prompts** that invite noisy judgments.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adopting/ablation-testing **single-dimension judging** vs multi-dimension judging for improved separability.
- [ ] Add a discussion/citation framing: **Pareto frontier between performance and reliability/hallucination** (especially for reasoning models).
- [ ] If relevant, add related-work paragraph on **benchmark builders** (not just benchmarks) for customizable evaluation.
- [ ] If GALILEO uses LLM-as-judge, compare to FURINA’s reported **dimension selection accuracy** and judge-human correlations; add a small calibration study.

## Quotes / details to potentially cite

- “a novel multi-agent collaboration pipeline that automatically constructs fully customizable RP benchmarks at any scale.” (abstract)
- “reasoning improves RP performance but simultaneously increases RP hallucinations” (abstract claim; verify exact wording in PDF if needed)
- Dimension set: Context Reliance, Factual Recall, Reflective Reasoning, Conversational Ability, Preference Alignment (Section 4.1)
