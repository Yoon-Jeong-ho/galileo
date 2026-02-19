# Revolutionizing Long-Term Memory in AI: New Horizons with High-Capacity and High-Speed Storage

- Year: 2026
- Venue: arXiv
- Authors: Hiroaki Yamanaka, Daisuke Miyashita, Takashi Toi, Asuka Maki, Taiga Ikeda, Jun Deguchi
- URL: https://arxiv.org/abs/2602.16192
- BibTeX key (if we add it): yamanaka2026stone
- Tags: memory, agents, experience-replay, storage-systems, retrieval

## One-sentence takeaway

A position/agenda paper arguing that agent memory should preferentially “store raw experiences, extract on demand” (STONE), plus (ii) aggregate across many experiences for probabilistic environments, and (iii) share experiences across agents—backed by simple toy/illustrative experiments.

## What problem does it solve?

- Claims current dominant agent-memory pipeline (“extract then store” summaries/rules) discards information that may become useful for future *different* tasks (latent learning / cross-task reuse).
- Highlights brittleness when using single retrieved experiences in stochastic settings (inter-context conflict) and inefficiency when each agent must collect its own experience via trial-and-error.

## What is the core method / protocol?

- Mostly conceptual; formalizes three paradigms:
  - **STONE (Store Then ON-demand Extract):** store full/raw experiences as memory; do task-specific extraction at retrieval/use time.
  - **Deeper insight discovery:** retrieve multiple relevant experiences and distill statistically (vs. using one “most relevant” episode).
  - **Experience memory sharing:** multiple agents contribute to a shared pool to amortize collection cost.
- Provides pseudo-algorithms contrasting “extract then store” vs STONE in a budgeted retrieval QA setting.

## What are the key metrics?

- In illustrative experiments:
  - QA with a limited “external retrieval budget” (counts number of times the agent must fetch external documents).
  - Multi-armed bandit average reward (ε-greedy vs naive replay).
  - HotpotQA success rate vs number of questions, comparing single-agent ExpeL vs “memory-sharing ExpeL”.

## What are the main results?

- **STONE vs extract-then-store:** storing whole documents reduces redundant refetching and preserves ability to answer later questions from the same source under a retrieval budget.
- **Deeper insight discovery:** trivial but clear demonstration that aggregating statistics (ε-greedy) beats “repeat last success / switch on failure” in stochastic bandits.
- **Memory sharing:** with 10 agents sharing trajectories/rules, reaches a target success rate with ~10x fewer questions per agent (as expected from parallelized collection).

## How is this similar to GALILEO?

- Motivates **non-parametric external memory** for agents and highlights the need for **cross-task reuse** of experience.
- Emphasizes retrieval + context injection patterns common in agentic systems.
- Calls out failure modes in stochastic environments when conditioning on a small number of conflicting memories—relevant to any system doing episodic retrieval.

## How is this different from GALILEO?

- Not a concrete algorithmic contribution; it is primarily a **systems/agenda** paper with simple experiments.
- Heavy emphasis on **storage hardware capacity/IOPS** and **KV-cache persistence** as “optimal AI memory form,” which may be orthogonal to GALILEO’s main novelty (depending on GALILEO’s modeling/algorithmic focus).
- Frames “comprehensive recall” as better served by sparse/logical search than ANN; may differ from GALILEO’s retrieval assumptions.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO provides a specific memory architecture or learning objective, it is likely stronger on:
  - rigorous ablations, benchmarks, and reproducible protocol;
  - concrete mechanisms for memory selection, compression, and retrieval;
  - principled handling of long-context constraints beyond “store everything”.

## Where GALILEO is weaker / needs to improve

- Consider whether GALILEO implicitly follows “extract then store” and thus may miss **latent future utility** of raw experiences.
- If GALILEO uses single-episode retrieval, it may be vulnerable to **stochasticity / inter-context conflict**; needs multi-episode aggregation or uncertainty-aware memory.
- If GALILEO is single-agent, this paper’s framing strengthens the case for **shared experience pools** (or at least multi-agent data collection) for faster improvement.

## Action items for GALILEO (experiments / method / writing)

- [ ] In related work, explicitly name and contrast **extract-then-store** vs **store-then-extract (STONE)**; position GALILEO on this axis.
- [ ] Add an experiment (or discussion) where information in an experience is **irrelevant to the current task** but becomes useful later (latent learning / cross-task transfer).
- [ ] If applicable, add a “stochastic environment” test where single retrieved memories conflict; evaluate **aggregation** strategies (retrieve-k + distill, majority vote, Bayesian/interval estimates, etc.).
- [ ] Discuss systems constraints: storage size, retrieval latency, and whether GALILEO could benefit from caching intermediate representations of long memories (e.g., persisted KV-cache or other reusable encodings).
- [ ] Consider a memory-sharing variant (multi-agent) or at minimum offline pooling of experiences across runs.

## Quotes / details to potentially cite

- Introduces the term **“Store Then ON-demand Extract: STONE”** as a contrast to “extract then store.”
- Key motivation: extraction-time task filtering can discard information useful for future different tasks (latent learning analogy).
- Notes “inter-context conflict” in probabilistic settings when injecting inconsistent info; argues for statistical extraction from multiple experiences rather than an effective learning rate ~1 from a single snippet.
- Challenges section provides a compact checklist: storage capacity, inference latency (quadratic attention; KV-cache reuse), comprehensive recall limitations of ANN for “retrieve all relevant”, ML for deeper insight, infrastructure for sharing, and privacy/security.
