# The ICASSP 2026 HumDial Challenge: Benchmarking Human-like Spoken Dialogue Systems in the LLM Era

- Year: 2026
- Venue: ICASSP 2026 (challenge summary paper; arXiv)
- Authors: Zhixian Zhao, Shuiyuan Wang, Guojian Li, Hongfei Xue, Chengyou Wang, Shuai Wang, Longshuai Xiao, Zihan Zhang, Hui Bu, Xin Xu, Xinsheng Wang, Hexin Liu, Eng Siong Chng, Hung-yi Lee, Lei Xie
- URL: https://arxiv.org/abs/2601.05564
- BibTeX key (if we add it): humdial2026zhao
- Tags: spoken-dialogue, benchmark, multi-turn, emotional-intelligence, full-duplex

## One-sentence takeaway

HumDial is an ICASSP 2026 shared task that benchmarks “human-like” spoken dialogue systems on (i) long-horizon emotional intelligence/empathetic response and (ii) full-duplex (listen-while-speaking) interaction/turn-taking, with a dataset derived from authentic human conversations.

## What problem does it solve?

- Lack of standardized, challenge-style evaluation for *human-like spoken dialogue* in the LLM / audio-LLM era.
- Specifically targets two hard-to-evaluate capabilities in natural spoken interaction:
  - long-term emotion understanding + empathetic generation
  - real-time, full-duplex turn-taking decisions under “listening while speaking”.

## What is the core method / protocol?

- Organizes a shared task (“HumDial Challenge”) with a common dataset and leaderboard-style evaluation.
- Two tracks:
  - **Emotional Intelligence**: evaluates long-term emotion understanding and empathetic generation.
  - **Full-Duplex Interaction**: evaluates real-time decision-making for turn-taking in full-duplex spoken dialogue.
- Paper is primarily a **task/dataset/results summary** rather than proposing a new learning algorithm.

## What are the key metrics?

- Not fully specified in the arXiv abstract; metrics appear track-specific:
  - Emotional intelligence: emotion understanding + empathy/response quality (likely a mixture of classification/sequence metrics + human/judge-based evaluation).
  - Full-duplex: turn-taking / timing decisions under full-duplex constraints.

## What are the main results?

- The paper reports “final results” for both tracks on the released dataset (details beyond the abstract require reading the full paper/HTML).

## How is this similar to GALILEO?

- Shared emphasis on **multi-turn interaction dynamics** and evaluation beyond single-turn accuracy.
- Potential neighbor in *protocol design*: benchmark framing, dataset from real interactions, and capability-specific tracks.

## How is this different from GALILEO?

- Focus is **spoken dialogue** (audio/omni models), emotion/empathetic behavior, and real-time turn-taking.
- Not centered on robustness-to-attack, drift, or adversarial multi-turn instability metrics (e.g., survival/time-to-failure) that GALILEO targets.

## Where GALILEO is stronger / cleaner (if true)

- GALILEO’s evaluation framing is more directly about **robustness/instability under multi-turn perturbations** with clearer failure-mode accounting (assuming survival / time-to-failure style reporting).

## Where GALILEO is weaker / needs to improve

- If GALILEO aims to generalize to naturalistic dialogue, it may be weaker on:
  - spoken interaction constraints (latency, turn-taking)
  - emotion/rapport/empathetic response evaluation.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider a short related-work paragraph positioning: “spoken full-duplex & long-horizon dialogue challenges exist; our contribution is robustness/instability measurement under multi-turn perturbations.”
- [ ] If we ever add an audio/spoken extension, HumDial’s **full-duplex** track is a useful reference point for evaluation design.

## Quotes / details to potentially cite

- “Achieving truly ‘human-like’ communication necessitates a dual capability: emotional intelligence … and robust interaction mechanisms … such as real-time turn-taking.”
- “We launched the first Human-like Spoken Dialogue Systems Challenge (HumDial) at ICASSP 2026 … across two tracks: (1) Emotional Intelligence … (2) Full-Duplex Interaction … under ‘listening-while-speaking’ conditions.”
