# AgentBench: Evaluating LLMs as Agents

- Year: 2024
- Venue: ICLR 2024 (arXiv)
- Authors: Xiao Liu, Hao Yu, Hanchen Zhang, Yifan Xu, Xuanyu Lei, Hanyu Lai, Yu Gu, Hangliang Ding, Kaiwen Men, Kejuan Yang, Shudan Zhang, Xiang Deng, Aohan Zeng, Zhengxiao Du, Chenhui Zhang, Sheng Shen, Tianjun Zhang, Yu Su, Huan Sun, Minlie Huang, Yuxiao Dong, Jie Tang
- URL: https://arxiv.org/abs/2308.03688
- BibTeX key (if we add it): liu2024agentbench
- Tags: agents, evaluation, benchmark, tool-use, multi-turn

## One-sentence takeaway

AgentBench is an early, broad benchmark suite for evaluating LLMs as interactive agents across code/game/web environments, highlighting that long-horizon instruction-following and decision-making (not just code ability) dominate failures.

## What problem does it solve?

- The community lacked a standardized, quantitative benchmark for *LLM-as-agent* behavior in interactive environments (multi-step decision-making, action execution), beyond narrow text-game settings or heavy multimodal simulators.
- Enables comparing closed and open models on “agentic” tasks and diagnosing common failure modes.

## What is the core method / protocol?

- Defines an evaluation framework for “LLM-as-Agent” and releases **8 environments** spanning:
  - **Code-grounded:** Operating System, Database, Knowledge Graph
  - **Game-grounded:** Digital Card Game, Lateral Thinking Puzzles
  - **Web-grounded:** House-holding, Web Shopping, Web Browsing
- Runs models in interactive loops (multi-round trajectories), scoring completion/success per environment and aggregating into an overall score.
- Includes analysis of *why* trajectories fail (e.g., instruction-following errors, repetition / getting stuck, poor long-term planning).

## What are the key metrics?

- Per-environment task success / completion scores (environment-specific).
- Aggregate / overall score across environments (paper also visualizes relative performance to the best model per environment).
- Diagnostic outcome categories (e.g., failures due to instruction-following vs planning/decision mistakes vs exceeding step/token limits).

## What are the main results?

- Large commercial models perform substantially better than many open-source models (including some up to ~70B), suggesting a gap in *agentic* alignment/robustness rather than only scale.
- Primary blockers to practical agent usability: **poor long-term reasoning, decision-making, and instruction following**.
- Improving instruction following and using higher-quality multi-round alignment data can improve agent performance.
- “More code training” is **not uniformly beneficial**: code-tuned models can help some agent tasks and hurt others (ambivalent impacts across environments).

## How is this similar to GALILEO?

- Shared emphasis on **multi-turn / long-horizon** behavior and robustness under interactive, sequential settings.
- Frames evaluation in terms of *agentic* success + common failure modes (getting stuck, repetition, instruction-following breakdowns), which overlaps with GALILEO-style reliability/consistency concerns.

## How is this different from GALILEO?

- AgentBench is a **broad benchmark suite** over diverse environments; it is less focused on the specific GALILEO axis (as opposed to, e.g., targeted stress tests / invariances / controlled perturbations depending on GALILEO’s exact setup).
- Evaluation looks largely like **task success across environments** rather than (for example) fine-grained behavioral invariants, controlled intervention studies, or explicit “drift / stability” measures.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO provides more controlled perturbations and clearer causal attribution of failure modes, it can complement AgentBench’s breadth with **cleaner experimental control**.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently lacks a broad “suite” of interactive environments, AgentBench suggests value in covering multiple domains (code + game + web) to avoid overfitting conclusions to one setting.

## Action items for GALILEO (experiments / method / writing)

- [ ] In related work, cite AgentBench as an early multi-environment *LLM-as-agent* benchmark and use its failure taxonomy (instruction-following, long-horizon planning, repetition/limit exceeded) to motivate GALILEO’s evaluation dimensions.
- [ ] Consider adding (or mapping to) at least one representative environment per category (code/game/web) to test generality.
- [ ] When discussing “tool use” vs “agentic reliability,” note their observation that code training has mixed effects across agent tasks.

## Quotes / details to potentially cite

- Abstract (benchmark + gap): “AgentBench, a multi-dimensional benchmark that consists of 8 distinct environments to assess LLM-as-Agent’s reasoning and decision-making abilities.”
- Abstract (failure modes): “poor long-term reasoning, decision-making, and instruction following abilities are the main obstacles for developing usable LLM agents.”
- Abstract (code training nuance): “training on code present ambivalent impacts on different agent tasks.”
