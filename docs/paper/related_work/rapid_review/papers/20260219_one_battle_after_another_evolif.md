# One Battle After Another: Probing LLMs’ Limits on Multi-Turn Instruction Following with a Benchmark Evolving Framework

- Year: 2025
- Venue: arXiv
- Authors: Qi Jia; Kaiwei Zhang; Xiujie Song; Ye Shen; Xiangyang Zhu; Guangtao Zhai
- URL: https://arxiv.org/html/2511.03508v1
- BibTeX key (if we add it): jia2025evolif (suggested)
- Tags: multi-turn, instruction-following, evolving-benchmark, user-simulation, robustness, survival-analysis

## One-sentence takeaway

Proposes an extensible “evolving” multi-turn instruction-following evaluation framework (with a patience-based stopping rule and process metrics) and instantiates it as EvolIF, showing large tier gaps in how long models can sustain faithful instruction following.

## What problem does it solve?

- Existing multi-turn instruction-following benchmarks are typically (i) short/fixed-length, (ii) static and thus saturate as models improve, and (iii) evaluated with endpoint-style metrics that miss interaction quality and “how long until failure”.
- The paper targets a more realistic evaluation of *sustained* instruction-following across topic switches and evolving/accumulating constraints.

## What is the core method / protocol?

- A benchmark *evolving framework* that separates:
  - **Surface form** (natural language phrasing) from
  - **Underlying intent state**, tracked via a **three-layer mechanism**:
    - **Topics** (multiple threads; can switch/backtrack)
    - **Instructions** per topic (current requirement set)
    - **Constraints** (atomic requirements; grouped to avoid contradictions)
- An **adaptive evaluation protocol** where the dialogue continues until a simulated user’s **patience** is exhausted (failures reduce patience), yielding variable-length conversations.
- A **process-oriented metric suite** to score the interaction across turns, not only the final turn.
- Instantiates this into **EvolIF** with **541 topics** and **9 constraint-type groups**; evaluates **10** frontier LLMs.

## What are the key metrics?

(Names described in the paper’s intro/overview.)

- **Continuity / conversational endurance**: how many turns a model sustains before “patience” termination.
- **Stability**: consistency of instruction-following across turns.
- **Recovery / error recovery rate**: ability to realign with intent after making a mistake.
- Additional “interaction process” quality metrics (paper mentions a suite; these three appear as central examples).

## What are the main results?

- Reports a clear tiering in sustained multi-turn instruction following.
- **GPT-5** is reported as best overall, with:
  - **18.54** average conversational turns sustained
  - **70.31%** robustness
  - outperforming **Gemini-2.5-Pro** by **11.41%** robustness (as stated in the abstract)
- **Error recovery is weak across all models**: even top models recover successfully **<30%** of the time (per intro summary).
- Constraint-type analysis: models struggle more with “global planning” constraints (e.g., precise length/keyword-count controls) than simpler local constraints (e.g., start-with/end-with).
- Sensitivity finding: top models can degrade noticeably under system-prompt or user-style variations.

## How is this similar to GALILEO?

- Same broad evaluation goal: characterizing *robustness over time* in multi-turn interaction under evolving requirements.
- Emphasizes **process metrics** (turn-by-turn behavior, persistence, recovery) rather than only end-task success.
- Introduces a **stopping-time** notion (patience) that pairs naturally with survival/time-to-failure analyses, which is often the right lens for “when does the system break?”.

## How is this different from GALILEO?

- This work is specifically framed around **instruction-following** with synthetic yet controlled constraint evolution; it is less about agentic tool-use or open-ended task execution (if those are central to GALILEO).
- It builds a *benchmark synthesis + evaluation* framework rather than focusing on a specific architecture/training method for improving multi-turn performance.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO’s tasks are grounded in real-world workflows (tool calls, documents, long-horizon objectives), it may offer higher ecological validity vs constraint-template instruction dialogues.
- If GALILEO has stronger anti-contamination controls / held-out distribution design, that could complement EvolIF’s “infinite stream” goal.

## Where GALILEO is weaker / needs to improve

- If GALILEO lacks a formal **patience / termination model** and **time-to-failure** reporting, it may miss an interpretable “upper limit” signal.
- If GALILEO doesn’t explicitly measure **error recovery**, this paper suggests it is a key shared failure mode worth instrumenting.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a “patience”-style adaptive stopping rule (or equivalent) so runs naturally produce variable-length dialogues and enable survival curves.
- [ ] Add process metrics aligned to: endurance, stability, and recovery (post-error realignment).
- [ ] Add constraint-type or requirement-class breakdowns, especially for “global planning” constraints (length/keyword counts) vs “local” constraints.
- [ ] Include prompt/user-style perturbation tests as a robustness dimension (system prompt variants; paraphrase/user persona shifts).

## Quotes / details to potentially cite

- “We propose an extensible framework for assessing multi-turn instruction-following ability… decouples linguistic surface forms from user intent simulation through a three-layer mechanism that tracks constraints, instructions, and topics.”
- “terminating a conversation only when the model exhausts a simulated user’s patience.”
- Abstract-reported headline numbers: GPT-5 average **18.54** turns; **70.31%** robustness; **+11.41%** robustness over Gemini-2.5-Pro.
