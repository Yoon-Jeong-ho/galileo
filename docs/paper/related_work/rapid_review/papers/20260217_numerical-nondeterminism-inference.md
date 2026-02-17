# Understanding and Mitigating Numerical Sources of Nondeterminism in LLM Inference

- Year: 2025
- Venue: arXiv
- Authors: Jiayi Yuan; Hao Li; Xinheng Ding; Wenya Xie; Yu-Jhe Li; Wentian Zhao; Kun Wan; Jing Shi; Xia Hu; Zirui Liu
- URL: https://arxiv.org/abs/2506.09501
- BibTeX key (if we add it): yuan2025numerical-nondeterminism-inference
- Tags: nondeterminism, reproducibility, inference, numerical-precision, evaluation, reasoning-models

## One-sentence takeaway

Even with **greedy decoding** and a fixed seed, low-precision inference (especially **BF16**) can produce meaningfully different outputs across hardware/runtime configs; they quantify this effect and propose **LayerCast** (store weights in 16-bit, compute in FP32) to get near-FP32 determinism at lower memory cost.

## What problem does it solve?

- Reproducibility of LLM evaluation/inference is often assumed when using greedy decoding (temperature=0), but real-world deployments vary in GPU type/count and batch size.
- For long-chain-of-thought / “reasoning” models, tiny early-token numerical differences can cascade into different reasoning traces, changing both **accuracy** and **output length**.

## What is the core method / protocol?

- Controlled study across:
  - Models: two reasoning models (DeepSeek-R1-Distill-Qwen-7B; DeepSeek-R1-Distill-Llama-8B) + two non-reasoning instruction models (Qwen2.5-7B-Instruct; Llama-3.1-8B-Instruct), plus extra appendix experiments.
  - Benchmarks: AIME’24, MATH500, LiveCodeBench (easy/medium/hard), plus appendix tasks.
  - Runtime configs: combinations of GPU type (e.g., L40S vs A100), GPU count (2 vs 4), and batch size (8/16/32), evaluated under different numeric formats (BF16/FP16/FP32).
- Explanation: nondeterminism arises from floating-point **non-associativity**; parallel kernels / reductions change operation order when runtime configuration changes, shifting logits slightly; when top-1 vs top-2 logit gaps are small, these shifts can flip the chosen token.
- Mitigation proposal (**LayerCast**): keep weights stored in 16-bit for memory, but upcast just-in-time so **matmul + compute** happen in FP32, aiming for FP32-like determinism without full FP32 memory overhead.

## What are the key metrics?

Greedy decoding (across multiple runtime configurations):
- **Std@Acc**: std. dev. of accuracy across runtime configs.
- **Avg_Std@Output_Length**: average per-example std. dev. in generated length across runtime configs.
- **Div_Index**: first token position where outputs diverge.
- **Avg_Std@top1_prob**: variability of top-1 token probability before divergence.

Random sampling:
- Pass@1 mean accuracy under multiple runs; plus std. dev. of Pass@1 across system configurations.

## What are the main results?

- Greedy decoding is *not* practically deterministic under BF16: for a reasoning model (DeepSeek-R1-Distill-Qwen-7B), they report up to **~9%** variation in accuracy across hardware/runtime settings, and up to **~9k tokens** difference in output length.
- Increasing precision reduces instability: FP16 helps; **FP32 is near-perfect** (almost zero variance in their tables).
- Divergence happens early and often in BF16 for reasoning models (many examples diverge across configs), while FP32 pushes divergence much later and makes it rare.
- Random sampling evaluations also inherit extra variance from numerical issues; BF16 often needs more trials to stabilize reported averages.
- **LayerCast** yields FP32-like determinism with lower memory footprint than full FP32 (they report sizable memory reduction, with very low divergence rates).

## How is this similar to GALILEO?

- Directly relevant to **multi-turn / long-horizon** evaluation: small early differences can cascade into different later behavior, which mirrors the “trajectory sensitivity” GALILEO cares about.
- Warns that observed “drift/instability” can be partly **systems-induced** (precision/hardware) rather than purely behavioral—important when reporting turn-of-failure / survival-style metrics.

## How is this different from GALILEO?

- Focus is on **numerical nondeterminism** in inference pipelines (hardware/runtime/precision), not social-pressure-induced drift, belief revision, or multi-turn persuasion protocols.
- Primary outcomes are reproducibility metrics (divergence point, variance across configs), not behavioral robustness under adversarial dialogue.

## Where GALILEO is stronger / cleaner (if true)

- GALILEO’s core contribution is a behavioral evaluation story (pressure vs evidence-driven revision controls, recovery dynamics, etc.) rather than systems-level determinism.

## Where GALILEO is weaker / needs to improve

- If GALILEO reports multi-turn instability metrics, it should explicitly control/report **numerical precision and runtime config**, otherwise “instability” could be confounded by inference nondeterminism.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a short **Reproducibility / Inference determinism** paragraph: specify numeric precision (BF16/FP16/FP32), GPU count/type, batch size, backend, and whether batch-invariant/deterministic kernels were used.
- [ ] If feasible, run a small ablation on one representative model showing how our multi-turn metrics change under BF16 vs FP16/FP32 (to bound the confound).
- [ ] When claiming improvements or comparing variants, consider reporting uncertainty that separates (a) sampling randomness from (b) system/precision variability.

## Quotes / details to potentially cite

- "[E]ven greedy decoding can yield different results across runs due to numerical precision issues." (Intro framing)
- They report (reasoning model, BF16, greedy): up to **~9%** accuracy variation and **~9,000 tokens** output-length differences across GPU count/type/batch size.
- Their diagnosis: floating-point **non-associativity** + parallel kernel reduction order changes + small top-1/top-2 probability gaps → token flip at a divergence index.
