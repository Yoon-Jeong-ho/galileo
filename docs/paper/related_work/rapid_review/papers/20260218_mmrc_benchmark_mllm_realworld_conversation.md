# MMRC: A Large-Scale Benchmark for Understanding Multimodal Large Language Model in Real-World Conversation

- Year: 2025
- Venue: arXiv
- Authors: Haochen Xue, Feilong Tang, Ming Hu, Yexin Liu, Qidong Huang, Yulong Li, Chengzhi Liu, Zhongxing Xu, Chong Zhang, Chun-Mei Feng, Yutong Xie, Imran Razzak, Zongyuan Ge, Jionglong Su, Junjun He, Yu Qiao
- URL: https://arxiv.org/abs/2502.11903
- BibTeX key (if we add it): xue2025mmrc
- Tags: multimodal, conversation, multi-turn, memory, benchmark, note-taking

## One-sentence takeaway

MMRC is a large-scale real-world multimodal conversation benchmark showing substantial multi-turn performance degradation in MLLMs and demonstrating that a simple externalized **note-taking** prompt strategy can mitigate several common failure modes.

## What problem does it solve?

- Existing MLLM evaluations under-measure *sustained*, open-ended, real-world-style multimodal conversations.
- Specifically targets whether models can (a) extract/update information across turns, (b) manage images, (c) recall earlier details, (d) reason over conversation state, and (e) refuse appropriately.

## What is the core method / protocol?

- Construct a benchmark of **5,120 multimodal conversations** with **28,720 manually labeled questions**.
- Evaluate six “core open-ended abilities”:
  - information extraction
  - multi-turn reasoning
  - information update
  - image management
  - memory recall
  - answer refusal
- Run evaluations on **20 MLLMs** and analyze typical degradation patterns over interaction.
- Mitigation: a **NOTE-TAKING strategy** that records key conversation information and provides reminders to the model during response generation (external memory / scratchpad-like summarization).

## What are the key metrics?

- Paper reports **accuracy** on the labeled questions, and emphasizes *accuracy drop during open-ended interactions*.
- (From abstract) analysis is framed via qualitative failure patterns rather than a single specialized survival/ToF metric.

## What are the main results?

- Across 20 MLLMs, performance drops as conversations become sustained/open-ended.
- Four common failure patterns identified:
  - long-term memory degradation
  - poor factual knowledge updates
  - accumulated assumptions / error propagation
  - reluctance to say “no” (refusal failures)
- NOTE-TAKING improves results across six evaluated MLLMs (claimed “significant performance improvements”).

## How is this similar to GALILEO?

- Shared focus on **multi-turn degradation** (memory drift / error propagation) and on explicitly measuring failures that only appear under sustained interaction.
- The “note-taking” mitigation is conceptually adjacent to **state tracking / intervention** strategies (externalized memory to stabilize behavior).

## How is this different from GALILEO?

- MMRC is primarily a **multimodal conversational capability benchmark**; GALILEO is centered on *pressure-driven belief drift vs evidence-driven revision* and related controls/metrics.
- MMRC’s headline mitigation is an external memory prompt strategy, not a diagnostic decomposition of drift vs revision (or recovery trajectories).

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO includes explicit *paired-control conditions* (pressure-only vs evidence) and *trajectory metrics* (time-to-failure, recovery, oscillation), it offers a cleaner causal story than “accuracy drop”.

## Where GALILEO is weaker / needs to improve

- If GALILEO does not already include an **external note-taking baseline**, MMRC suggests it as a simple, practical mitigation that could be a strong reference point.
- If GALILEO is text-only, MMRC highlights that multimodal settings may exacerbate state-management and memory issues.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a simple “NOTE-TAKING” baseline: maintain a running bullet summary of key facts per turn and prepend/append it to each response (or provide it as separate context) to test whether flips/degradation reduce.
- [ ] When discussing degradation, cite MMRC’s failure taxonomy as supporting evidence that long-horizon interaction introduces memory/update/refusal pathologies.

## Quotes / details to potentially cite

- “MMRC comprises 5,120 conversations and 28,720 corresponding manually labeled questions…”
- “We identify four common failure patterns: long-term memory degradation, inadequacies in updating factual knowledge, accumulated assumption of error propagation, and reluctance to say no.”
- “We propose a simple yet effective NOTE-TAKING strategy… enhancing conversational capabilities.”
