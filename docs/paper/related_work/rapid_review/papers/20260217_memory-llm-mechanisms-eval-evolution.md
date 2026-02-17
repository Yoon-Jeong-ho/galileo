# Memory in Large Language Models: Mechanisms, Evaluation and Evolution

- Year: 2025
- Venue: arXiv
- Authors: Wendong Li; Kani Song; Jiaye Lu; Gang Li; Liuchun Yang; Sheng Li
- URL: https://arxiv.org/abs/2509.18868
- BibTeX key (if we add it): memoryLLMs2025li
- Tags: memory, evaluation, governance, RAG, long-context, model-editing, benchmarking

## One-sentence takeaway

A survey/framework paper that proposes a unified definition + taxonomy of “LLM memory” and a layered, setting-controlled evaluation protocol spanning parametric/contextual/external/procedural memory, with a governance loop for updating/forgetting.

## What problem does it solve?

- “Memory” is used inconsistently across parametric recall, long-context behavior, retrieval-augmented setups, and cross-session/episodic persistence, causing non-comparable results and unclear deployment guidance.
- Evaluations often conflate capability with information availability (e.g., RAG with different corpora/timelines) and conflate retrieval quality with generation faithfulness/attribution.
- Practical needs: auditing timeliness, leakage, outdated answers, and update/forget workflows.

## What is the core method / protocol?

- Operational definition: LLM memory = persistent state written during pretraining/finetuning/inference that can later be addressed and stably influences outputs.
- Taxonomy: (1) parametric, (2) contextual (working memory/long-context effects), (3) external (non-parametric stores, e.g., RAG), (4) procedural/episodic.
- “Memory quadruple”: location, persistence, write/access path, controllability.
- Causal chain linking mechanism↔evaluation↔governance: **write → read → inhibit/update**.
- **Three-setting parallel protocol** to make evaluations comparable:
  - parametric-only (closed-book)
  - offline retrieval
  - online retrieval
  (Goal: decouple model capability from information availability on the same data slice/timeline.)
- Layered evaluation proposal:
  - Parametric: closed-book recall; pre/post-edit differential; memorization & privacy risk.
  - Contextual: position–performance curves; “mid-sequence drop” / lost-in-the-middle behavior.
  - External: correctness vs snippet-level attribution/faithfulness (two-channel eval: retrieval × faithfulness).
  - Procedural/episodic: cross-session consistency; timeline replay (mentions E-MARS+).
- Governance proposal (DMM-Gov): auditable loop coordinating data adaptation (DAPT/TAPT), PEFT, model editing (ROME/MEND/MEMIT/SERAC), and RAG with monitoring/rollback/change audits.

## What are the key metrics?

- Not a single benchmark; proposes metric *families* by memory type:
  - Closed-book recall / factual accuracy (parametric-only)
  - Edit success/side-effects via **edit differentials** (pre/post editing)
  - Long-context positional robustness (position curves; mid-sequence drop)
  - External memory: answer correctness **and** snippet-level attribution/faithfulness (separating retrieval quality from generation faithfulness)
  - Cross-session consistency & timeline replay for procedural/episodic memory
- Also stresses statistical hygiene: inter-rater agreement, paired tests, multiple-comparison correction.

## What are the main results?

- Primary contribution is a unifying conceptual + evaluation + governance framework (not SOTA numbers).
- Provides concrete, testable propositions and a recommended “minimal evaluation card” concept for memory systems.

## How is this similar to GALILEO?

- Aligns with GALILEO’s interest in **multi-turn / longitudinal behavior** and separating *capability* from *context/retrieval availability*.
- The “timeline replay / cross-session consistency” framing overlaps with evaluating stability over interaction trajectories.

## How is this different from GALILEO?

- Focuses broadly on “memory” (parametric, RAG, editing, governance) rather than GALILEO’s more specific robustness phenomena (e.g., multi-turn drift / pressure / instability) as a primary object.
- Emphasizes governance/auditing and update/forget pipelines as first-class outputs.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO has tightly controlled multi-turn protocols and clear robustness metrics, it may be more *operationally crisp* than a broad survey framework.

## Where GALILEO is weaker / needs to improve

- If GALILEO discusses “memory-like” effects (context accumulation, trajectory dependence) without a strict setting protocol, this paper’s **three-setting decoupling** could strengthen claims.
- Might need better separation between retrieval quality vs generation faithfulness when tools/RAG are involved.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adopting (or citing) the **three-setting parallel protocol** idea whenever comparing systems with different access to external info.
- [ ] For any tool/RAG variants, report a **two-channel evaluation**: retrieval quality × response faithfulness/attribution.
- [ ] Add a short “memory taxonomy” paragraph to related work: parametric vs contextual vs external vs episodic, and place GALILEO explicitly.
- [ ] If doing long-context studies, add **position–performance curves** / lost-in-the-middle diagnostics.

## Quotes / details to potentially cite

- Operational definition: memory is a “persistent state … written during pretraining, finetuning, or inference … subsequently addressed … [that] stably influences outputs.”
- Proposed three-setting protocol: parametric-only vs offline retrieval vs online retrieval to avoid distorted comparisons.
- External memory evaluation should decouple correctness from snippet-level attribution/faithfulness (retrieval × faithfulness).
