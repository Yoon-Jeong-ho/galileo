# ThinkBench: Dynamic Out-of-Distribution Evaluation for Robust LLM Reasoning

- Year: 2025
- Venue: arXiv
- Authors: Shulin Huang, Linyi Yang, Yan Song, Shuang Chen, Leyang Cui, Ziyu Wan, Qingcheng Zeng, Ying Wen, Kun Shao, Weinan Zhang, Jun Wang, Yue Zhang
- URL: https://arxiv.org/abs/2502.16268
- BibTeX key (if we add it): thinkbench_huang_2025
- Tags: dynamic-eval, ood, data-contamination, reasoning, robustness, benchmark

## One-sentence takeaway

ThinkBench evaluates LLM reasoning more robustly by dynamically generating OOD variants (semi-factual scenario/attack perturbations) to reduce contamination/leakage and expose large ID→OOD performance gaps.

## What problem does it solve?

- Static benchmarks for reasoning are increasingly unreliable due to (i) training/test contamination and (ii) answer leakage/memorization, which can inflate apparent reasoning ability.
- Need a way to measure *generalization* in reasoning, not just recall of benchmark items.

## What is the core method / protocol?

- Proposes **dynamic OOD dataset generation** grounded in a “semi-factual” perturbation idea (they reference causal/semi-factual causality framing), with two granularity levels:
  - **Scenario-level**: modify broader context/scenario elements.
  - **Attack-level**: modify more local elements (textual details) to create harder, less-memorized variants.
- Constructs an OOD evaluation set of **2,912 samples** derived from reasoning tasks (paper mentions AIME-500, AIME 2024, GPQA Diamond as sources).
- Evaluates a mix of **reasoning models** (test-time compute/search + PRM) and **non-reasoning models** under “identical experimental conditions”, and also evaluates **PRMs** with best-of-n style decoding.

## What are the key metrics?

- Accuracy (ID vs OOD), with emphasis on **performance decay / gap** from in-distribution to dynamically generated OOD variants.
- Secondary: comparisons under different test-time compute budgets (best-of-n) and different PRMs.

## What are the main results?

- Most evaluated models show **substantial ID→OOD drops**, suggesting non-robust reasoning and/or contamination effects on standard (ID) splits.
- Paper reports average performance decay of **~24.9% on AIME-500** and **~11.8% on AIME 2024** across models (as described in the intro).
- Stronger reasoning models (e.g., o1/o3-style, DeepSeek-R1, s1 as mentioned) maintain higher absolute accuracy but still face measurable OOD gaps.

## How is this similar to GALILEO?

- Shared theme: **robust evaluation** beyond surface performance, focusing on whether models truly generalize vs. exploit artifacts/leakage.
- The “dynamic generation” framing is conceptually aligned with creating **harder, less-leaky variants** of evaluation prompts to probe robustness.

## How is this different from GALILEO?

- ThinkBench targets **math/science reasoning benchmarks** (AIME/GPQA) and is framed around *contamination/leakage* in public benchmark data.
- GALILEO’s focus (as positioned in this repo) is not primarily about benchmark contamination; it emphasizes robustness in its own task setting and multi-round/behavioral outcomes rather than math-problem OOD.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO evaluations are fully **self-contained / newly generated / controlled**, it may have less exposure to the specific “public benchmark memorization” issue ThinkBench addresses.
- If GALILEO uses deterministic pipelines and tracked artifacts, the auditability/reproducibility story may be cleaner than “dynamic generation at evaluation time” unless the generator is frozen and logged.

## Where GALILEO is weaker / needs to improve

- If GALILEO relies on relatively static prompt sets, it may benefit from a more explicit **ID vs OOD** stress-test methodology (systematically perturbed variants) to quantify robustness gaps.
- Need to ensure any dynamic variant generation is **logged and reproducible** (seeded transforms; stored generated items) to avoid evaluation drift.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a short related-work paragraph citing ThinkBench as evidence that **static evaluations overestimate capability** due to contamination, motivating robustness checks.
- [ ] Prototype a “ThinkBench-style” OOD split for one GALILEO task family: define a small set of **structured perturbations** (scenario-level vs attack-level analogs) and report Δ metrics (ID→OOD).
- [ ] If adopting dynamic generation, ensure: (i) generator code is versioned, (ii) outputs are saved as artifacts, (iii) seeds/configs are recorded.

## Quotes / details to potentially cite

- “ThinkBench proposes a dynamic data generation method for constructing out-of-distribution (OOD) datasets…” (abstract)
- OOD set size: “2,912 samples” (abstract/intro)
- Reported average performance decay: “24.9% … on AIME-500” and “11.8% … on AIME 2024” (intro text in arXiv HTML)
