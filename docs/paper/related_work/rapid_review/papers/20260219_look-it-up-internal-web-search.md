# Look It Up: Analysing Internal Web Search Capabilities of Modern LLMs

- Year: 2025
- Venue: arXiv
- Authors: Sahil Kale et al.
- URL: https://arxiv.org/abs/2511.18931
- BibTeX key (if we add it): lookitup2025
- Tags: web-search, tool-use, evaluation, calibration, cost

## One-sentence takeaway

A benchmark + analysis showing that built-in “internal web search” in closed models boosts factual accuracy but often worsens confidence calibration and still fails frequently due to weak query formulation, especially on post-cutoff “current” questions.

## What problem does it solve?

- We lack externally-measurable evaluation of *when* closed-source LLMs decide to invoke their integrated web search tools, and whether invocation is efficient/beneficial.
- Prior web agent benchmarks often assume browsing/search is attempted; this instead targets *invocation calibration* and cost/accuracy tradeoffs for native search.

## What is the core method / protocol?

- Build a 2-part benchmark:
  - **Static split (783 questions):** pre-cutoff, temporally anchored factual QA; run each question twice (search disabled vs enabled) and collect an explicit confidence score.
  - **Dynamic split (288 questions):** post-cutoff, rewritten to require “current” answers (e.g., “Who is the current CEO …”); search enabled with up to 2 calls.
- Evaluate two commercial models with native search tools (GPT-5-mini, Claude Haiku 4.5) under temperature 0.
- Metrics focus on:
  - accuracy deltas from search,
  - invocation rates and number of calls,
  - outcome transitions (correct→correct, incorrect→correct, etc.),
  - calibration (ECE, Brier) of stated confidence,
  - cost per improvement (web-call cost abstraction).

## What are the key metrics?

- Accuracy (exact match against sets of acceptable answers)
- Search invocation rate; calls/query (0/1/2)
- Outcome transition categories (beneficial vs harmful)
- Calibration: ECE, Brier score (static split, with/without search)
- Cost metrics: avg extra cost/query; cost per strict improvement

## What are the main results?

- **Static split:** search substantially improves accuracy for both models (e.g., GPT-5-mini ~0.52→~0.85; Claude ~0.41→~0.75).
- **Dynamic split:** both models invoke search frequently (~88–91% of queries) but remain **<70% accuracy**, with failures attributed to **poor query formulation / inability to recover after a bad first search**.
- **Calibration degrades with search:** even as accuracy rises, models become more **overconfident** when search is available (ECE/Brier worsen).
- **Diminishing returns from multiple calls:** two calls often correlate with lower accuracy (indicative of “flailing” after an unhelpful first retrieval).
- **Cost:** web calls are cheap on average, but cost-per-improvement depends on model and split.

## How is this similar to GALILEO?

- Shares the broader theme of **evaluating multi-step tool use** and **reliability** under interactive settings.
- Highlights a key issue relevant to agentic systems: **tool access can change behavior** (confidence, stability) in non-obvious ways.

## How is this different from GALILEO?

- Focused specifically on **native/internal web search tools** in closed-source LLM APIs, and largely on **short factual QA**.
- Not primarily about long-horizon agent planning, complex workflows, or robust multi-turn task completion beyond 1–2 tool calls.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO is about agent protocols/robustness: this paper suggests a clearer story for GALILEO around **failure recovery after tool mistakes**, beyond one-shot retrieval.
- Opportunity to position GALILEO as addressing **multi-step robustness** rather than just “does retrieval help”.

## Where GALILEO is weaker / needs to improve

- This work directly measures **calibration effects** (confidence vs correctness) when tools are available; if GALILEO does not measure confidence/uncertainty or “should I call the tool?”, that may be a gap.

## Action items for GALILEO (experiments / method / writing)

- [ ] In related work: cite as evidence that **tool access increases accuracy but can worsen calibration/overconfidence**, motivating robust agent protocols.
- [ ] Consider adding an analysis slice: after a failed tool call, do agents recover or “spiral”? (analogous to the observed 2-call degradation).
- [ ] If we report agent self-assessments, check calibration *with vs without* tools.

## Quotes / details to potentially cite

- Abstract-level framing: internal web search “meaningfully improves factual accuracy” but models remain “overconfident”, sometimes “skip retrieval when it is essential”, and “falter once initial search queries underperform”.
- Dataset sizes: static 783 (pre-cutoff), dynamic 288 (post-cutoff), max 2 web calls.
- Key empirical claim: dynamic accuracy remains below ~70% due to weak query formulation.
