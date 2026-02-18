# A Personalized Conversational Benchmark: Towards Simulating Personalized Conversations

- Year: 2025
- Venue: arXiv
- Authors: Li Li; Peilin Cai; Ryan A. Rossi; Franck Dernoncourt; Branislav Kveton; Junda Wu; Tong Yu; Linxin Song; Tiankai Yang; Yuehan Qin; Nesreen K. Ahmed; Samyadeep Basu; Subhojyoti Mukherjee; Ruiyi Zhang; Zhengmian Hu; Bo Ni; Yuxiao Zhou; Zichao Wang; Yue Huang; Yu Wang; Xiangliang Zhang; Philip S. Yu; Xiyang Hu; Yue Zhao
- URL: https://arxiv.org/abs/2505.14106
- BibTeX key (if we add it): personaconvbench2025
- Tags: personalization, multi-turn, benchmark, conversation-graphs, reddit

## One-sentence takeaway

PersonaConvBench is a large Reddit-derived benchmark that *jointly* tests personalization + multi-turn conversational structure via classification, regression, and response-generation tasks on graph-structured conversations.

## What problem does it solve?

- Existing personalization benchmarks often ignore multi-turn conversational structure (treat utterances as independent), while multi-turn dialogue benchmarks are often user-agnostic.
- The paper proposes a single benchmark to study how *user history / personalized context* affects LLM behavior in realistic, multi-user, branching conversations.

## What is the core method / protocol?

- Data: Reddit conversations across **10 domains**.
- Representation: a conversation as a **graph** (reply structure), with user-specific **trajectories** extracted from the graph.
- For a target user reply, construct prediction instances using:
  - the earlier part of the same trajectory (conversation context)
  - plus additional **user history** (other conversations by the same user)
- Tasks:
  1) **Personalized conversational sentiment classification** (label a message / response sentiment given convo + user history)
  2) **Personalized conversational impact forecasting** (regress future impact/feedback score)
  3) **Personalized follow-up text generation** (generate the user’s next response)
- Evaluation setting includes a temporal split idea (train/test over time) to reflect realistic deployment.
- Baselines: a unified prompting setup for a mix of commercial + open LLMs; compare with and without conversational/personal history.

## What are the key metrics?

- Classification: standard classification metrics (reported as classification performance; likely accuracy/F1).
- Regression: standard regression error metrics (reported as forecasting performance).
- Generation: automatic text-generation similarity metrics (reported for follow-up response generation).

(Need to check the PDF for exact metric list if we want to cite numbers beyond the headline gain.)

## What are the main results?

- Including personalized conversational history can significantly improve results versus non-conversational / non-personalized baselines.
- Headline: **198% relative gain** over the best non-conversational baseline in sentiment classification (under their prompting setup).
- They also highlight that irrelevant history can hurt, motivating careful personalization (history selection).

## How is this similar to GALILEO?

- Shared theme: **multi-turn evaluation** where prior turns/history materially change outcomes.
- Reinforces the general claim that *context selection matters* and that multi-turn setups can reveal failures/variance not visible in single-turn evaluations.

## How is this different from GALILEO?

- Primary objective is **personalization** (user-specific style/history) rather than robustness to pressure, drift-vs-revision controls, or recovery dynamics.
- Focuses on predictive tasks (sentiment / impact) and response generation, not explicit “belief stability under pressure” or time-to-failure robustness.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO centers controlled multi-turn perturbations (pressure/evidence) and stability metrics, it is more diagnostic for **drift/flip dynamics** than a broad personalization benchmark.

## Where GALILEO is weaker / needs to improve

- If GALILEO lacks a strong personalization axis, this paper is a reminder that “history” can mean **user identity/history** in addition to conversation-local context.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a brief related-work paragraph noting that personalization benchmarks (e.g., PersonaConvBench) treat multi-turn history as a *signal to exploit*, whereas GALILEO treats multi-turn interaction as a *stress test* for stability/robustness.
- [ ] Consider an ablation: “same conversation history length, but swap user history (same-domain other-user)” to quantify when *personal* history vs *generic* history causes behavior shifts.

## Quotes / details to potentially cite

- Benchmark scope claim: “19,215 posts … over 111,239 conversations from 3,878 users” (from the HTML abstract/intro).
- High-level benchmark framing: integrates personalization + conversational structure; 3 task types (classification/regression/generation) across 10 domains.
- Headline improvement: “198% relative gain … in sentiment classification” when adding personalized conversational history.
