# IDRBench: Interactive Deep Research Benchmark

- Year: 2026
- Venue: arXiv
- Authors: Yingchaojie Feng, Qiang Huang, Xiaoya Xie, Zhaorui Yang, Jun Yu, Wei Chen, Anthony K. H. Tung
- URL: https://arxiv.org/abs/2601.06676
- BibTeX key (if we add it): Feng2026IDRBench
- Tags: agents, deep-research, interactive, evaluation, multi-turn

## One-sentence takeaway

IDRBench proposes an evaluation setup for *interactive* deep-research agents that explicitly measures the benefit of asking the user for guidance versus the interaction cost (turns/tokens).

## What problem does it solve?

- Existing “deep research agent” benchmarks mostly assume:
  - user intent is fully specified up front, and
  - evaluation only cares about the final report.
- In real deep research, goals are underspecified and evolve; an agent that never checks back can silently drift or hallucinate intent.
- Prior benchmarks do not (a) model dynamic user feedback, nor (b) quantify the *overhead* of interaction.

## What is the core method / protocol?

- Introduces **IDRBench**, positioned as a benchmark for **interactive deep research**.
- Key ingredients (as described in abstract/intro):
  - a **modular multi-agent research framework** with **on-demand interaction**
  - a **reference-grounded user simulator** intended to scale evaluation
  - an **interaction-aware evaluation suite** measuring:
    - benefits: quality + alignment
    - costs: turns + tokens

## What are the key metrics?

- **Quality** of research output (reference-grounded evaluation; details not fully parsed in this rapid pass).
- **Alignment** to user intent / preferences, in an interactive setting.
- **Interaction cost**:
  - number of interaction turns
  - number of tokens (proxy for cost/latency).
- The framing emphasizes the *trade-off curve* between benefit and overhead.

## What are the main results?

- Across **7 SOTA LLMs**, adding interaction:
  - “consistently improves research quality and robustness”
  - can “outweigh differences in model capacity”
  - reveals trade-offs in interaction efficiency (some models need more/less interaction for similar gains).

## How is this similar to GALILEO?

- Shares the core thesis that **multi-turn dynamics matter** and that evaluating only end outputs can miss important failure modes (drift, misalignment).
- Provides a concrete example of evaluating *policies over turns* (when to ask / how much to ask), not just single-shot answers.

## How is this different from GALILEO?

- Focus is **interactive deep research** (web exploration + long-form synthesis), rather than (presumably) GALILEO’s focus on conversational robustness/consistency under user pressure or multi-turn perturbations.
- Relies on a **user simulator** + reference-grounded scoring; GALILEO may emphasize adversarial users/pressure, stance flips, or robustness-to-manipulation metrics.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO targets *robustness under adversarial pressure* (sycophancy / persuasion / stance flips), it may provide clearer stress-test axes than a broad deep-research setting.
- GALILEO likely has more direct operationalization of “bad behaviors” (e.g., flip rates, time-to-failure), whereas IDRBench is broader and may entangle many components (search, synthesis, citation quality, etc.).

## Where GALILEO is weaker / needs to improve

- GALILEO writing/eval may need to more explicitly quantify **interaction cost** (turns/tokens) as a first-class metric and show Pareto trade-offs.
- If GALILEO assumes static intent, it may miss realistic *goal-evolution* and *clarification* phenomena.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a short “**interaction trade-off**” subsection: benefit metrics vs turns/tokens (even if only as an analysis plot).
- [ ] Consider a small ablation: **no-clarification** vs **on-demand clarification** policy, measuring downstream robustness.
- [ ] In related work, cite IDRBench as evidence that **interaction is an evaluable dimension** (not just UX).

## Quotes / details to potentially cite

- “In practice, research goals are often underspecified and evolve during exploration, making sustained interaction essential for robust alignment.”
- “IDRBench … jointly measures interaction benefits (quality and alignment) and costs (turns and tokens).”
- “Interaction consistently improves research quality and robustness, often outweighing differences in model capacity, while revealing substantial trade-offs in interaction efficiency.”
