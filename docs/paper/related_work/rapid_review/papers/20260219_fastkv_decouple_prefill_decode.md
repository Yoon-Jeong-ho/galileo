# FastKV: Decoupling of Context Reduction and KV Cache Compression for Prefill-Decoding Acceleration

- Year: 2025
- Venue: arXiv
- Authors: Dongwon Jo; Jiwon Song; Yulhwa Kim; Jae-Joon Kim
- URL: https://arxiv.org/abs/2502.01068
- BibTeX key (if we add it): fastkv2025jo
- Tags: long-context, inference, kv-cache, compression, prefill, decoding, token-pruning

## One-sentence takeaway

FastKV accelerates **both** long-context prefill and decoding by (i) doing full-context compute only up to a mid-layer **Token-Selective Propagation (TSP)** point, then propagating only salient tokens, and (ii) **decoupling** this prefill context reduction from a separate, layer-wise **KV retention** budget used for decoding.

## What problem does it solve?

- Long-context LLM inference is bottlenecked by:
  - **Prefill**: attention compute grows (roughly) quadratically with prompt length.
  - **Decoding**: KV cache size/bandwidth grows linearly with prompt length and dominates latency/throughput.
- Prior work often accelerates **only decoding** (KV eviction after full prefill) or accelerates prefill by pruning early, but then **couples** the prefill reduction and the decoding KV budget, causing accuracy drops.

## What is the core method / protocol?

- Key empirical motivation: “important tokens” (high-attention mass) vary a lot in **early** layers, but become more **stable** in later layers.
- **Two-stage prefill**:
  - Run **full-context** computation up to a chosen **TSP layer**.
  - At the TSP layer, select top-K “informative” tokens and **propagate only those hidden states** to later layers (Token-Selective Propagation).
- **Decoupled KV compression**:
  - From the propagated tokens, each layer selects a subset of KV entries to retain for decoding (layer-wise KV retention).
  - Crucially: TSP “propagation rate” (prefill compute) and KV “retention rate” (decoding budget) are **independently tunable**.

## What are the key metrics?

- Prefill speed / latency vs full-context baseline.
- Decoding speed / latency vs full-context baseline.
- Accuracy / task performance (they cite LongBench; claim within ~1% drop).

## What are the main results?

- Up to **1.82×** prefill speedup and **2.87×** decoding speedup vs full-context, while matching (or nearly matching) the accuracy of baselines that only accelerate decoding.
- Compared against representative methods (as described in intro): StreamingLLM, SnapKV (decode-focused) and GemFilter, PyramidInfer (prefill-aware).

## How is this similar to GALILEO?

- If GALILEO’s setting includes **long-context transformer inference/training** constraints, FastKV is relevant as a concrete, layer-aware way to trade compute/memory vs fidelity.
- More generally, it is a clean example of a design pattern: **separate (decouple) the knob that reduces upstream compute** from the knob that controls downstream memory/throughput.

## How is this different from GALILEO?

- FastKV is an **inference-time systems/method** paper focused on transformer internal efficiency (prefill + KV cache), not a task/benchmark/protocol paper.
- Uses attention-derived token “importance” to prune/propagate, rather than task-driven selection.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO emphasizes evaluation protocols, robustness, or application-centric claims, FastKV is narrower: it optimizes inference mechanics and reports primarily speed/accuracy trade-offs.

## Where GALILEO is weaker / needs to improve

- If GALILEO relies on long contexts, it may need a clearer accounting of **prefill vs decoding** costs (and possibly a method knob for each), similar to FastKV’s explicit separation.

## Action items for GALILEO (experiments / method / writing)

- [ ] If we report long-context inference costs, split results into **prefill latency** and **decode throughput/latency**, not just an aggregate.
- [ ] Consider adding a short related-work paragraph framing: prior KV-compression work often leaves prefill slow; prefill-aware pruning can hurt accuracy due to early-layer instability; FastKV-style **late-layer context stabilization** motivates delaying pruning.

## Quotes / details to potentially cite

- “Recent works that compress KV caches with prefill acceleration … inadvertently tie the prefill compute reduction to the decoding KV budget.”
- FastKV key idea: “perform full-context computation until a Token-Selective Propagation (TSP) layer, which forwards only the most informative tokens to subsequent layers,” and then “independently selects salient KV entries for caching, thereby decoupling KV budget from the prefill compute reduction.”
- Reported speedups: “up to 1.82× in prefill and 2.87× in decoding … maintaining accuracy drop within 1% on LongBench.”
