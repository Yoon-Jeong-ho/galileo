# MARS-Bench: A Multi-turn Athletic Real-world Scenario Benchmark for Dialogue Evaluation

- Year: 2025
- Venue: EMNLP 2025 Findings (arXiv)
- Authors: Chenghao Yang, Yinbo Luo, Zhoufutu Wen, Qi Chu, Tao Gong, Longxiang Liu, Kaiyuan Zhang, Jianpeng Jiao, Ge Zhang, Wenhao Huang, Nenghai Yu
- URL: https://arxiv.org/abs/2505.23810
- BibTeX key (if we add it): marsbench2025yang
- Tags: multi-turn, dialogue, evaluation, benchmark, cross-turn-dependency, long-context

## One-sentence takeaway

MARS-Bench is a realistic, long-horizon multi-turn dialogue benchmark (from play-by-play sports commentary) designed to stress-test cross-turn dependency and “motivation transfer”, showing substantial degradation—especially for open models—and benefits from explicit reasoning.

## What problem does it solve?

- Existing dialogue benchmarks under-stress *long, complex* sessions where the user goal/motivation can shift and later turns depend on earlier details.
- Lack of a benchmark that simultaneously targets (i) very long dialogues, (ii) interactive multi-turn behavior, and (iii) cross-turn tasks requiring retrieval/composition across turns.

## What is the core method / protocol?

- Construct dialogues from play-by-play athletic text commentary to get “real-world scenario” structure.
- Benchmark is organized around three aspects (as stated in the abstract):
  - **Ultra Multi-turn** (very long sessions)
  - **Interactive Multi-turn** (requires turn-by-turn adaptation)
  - **Cross-turn Tasks** (explicit cross-turn dependency)
- Run a suite of LLMs; compare closed vs open models.
- Analyze performance changes and provide a mechanistic/interpretability angle: attention visualization in Qwen2.5-7B-Instruct suggests performance degradation linked to **attention sinks** caused by special tokens.

## What are the key metrics?

- Not fully specified from the arXiv abstract page alone.
- Likely task success / accuracy-style metrics per task category + breakdowns by turn position / dialogue length; confirm from PDF when needed.

## What are the main results?

- Closed-source LLMs “significantly outperform” open-source alternatives on these long, complex multi-turn settings.
- **Explicit reasoning** “significantly boosts” robustness on long complex dialogues.
- Models struggle with **motivation transfer** and **sophisticated cross-turn dependency**.
- Mechanistic finding: **special-token attention sinks** may contribute to degradation (shown via attention visualization on Qwen2.5-7B-Instruct).

## How is this similar to GALILEO?

- Shared focus on *multi-turn* robustness/quality under realistic interaction patterns.
- Emphasis on failure modes that appear only over longer horizons and with cross-turn dependencies.

## How is this different from GALILEO?

- Domain is sports/athletics scenarios derived from commentary; GALILEO (as positioned in our related-work cluster) is broader and/or centered on robustness phenomena like drift, pressure, sycophancy, jailbreak trajectories, etc.
- MARS-Bench seems more like a **task/benchmark suite** for dialogue evaluation than a targeted robustness-protocol paper (based on abstract).
- Includes a specific mechanistic interpretation claim (attention sinks due to special tokens) rather than only black-box behavioral evaluation.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO provides clearer causal/protocol separation (e.g., drift vs evidence-driven revision; pressure vs information), it may yield more interpretable comparisons than a domain-bound benchmark.
- If GALILEO has standardized time-to-failure / survival-style metrics, it may connect robustness to a unified quantitative framing.

## Where GALILEO is weaker / needs to improve

- Might need more “real-world” dialogue sources and interactive tasks with authentic cross-turn structure (MARS-Bench’s construction angle is a good example).
- Mechanistic analysis could be a gap if GALILEO stays purely behavioral.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a related-work paragraph: long-horizon dialogue benchmarks that stress cross-turn dependency; cite MARS-Bench as evidence that explicit reasoning helps in complex multi-turn sessions.
- [ ] Consider a small ablation: evaluate whether special-token formatting (system prompts / separators) affects multi-turn degradation in our setting (tie-in to “attention sink” hypothesis).

## Quotes / details to potentially cite

- From the arXiv abstract: benchmark targets “Ultra Multi-turn, Interactive Multi-turn, and Cross-turn Tasks” and highlights weaknesses in “motivation transfer” and “sophisticated cross-turn dependency”.
- From the arXiv abstract: “explicit reasoning significantly boosts LLMs' robustness” on long complex dialogues; “attention sinks due to special tokens” linked to degradation (attention visualization in Qwen2.5-7B-Instruction).
