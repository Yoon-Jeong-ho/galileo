# Defensive M2S: Training Guardrail Models on Compressed Multi-turn Conversations

- Year: 2026
- Venue: arXiv
- Authors: Hyunjun Kim
- URL: https://arxiv.org/abs/2601.00454
- BibTeX key (if we add it): kim2026defensive_m2s
- Tags: safety, guardrails, multi-turn, compression, jailbreaks, efficiency

## One-sentence takeaway

Train guardrail classifiers on a deterministic “multi-turn → single-turn” compressed representation of conversations to drastically cut tokens (≈95%) while improving multi-turn jailbreak detection recall.

## What problem does it solve?

- Multi-turn safety screening is expensive because guardrail models must ingest long conversation histories (high latency/cost).
- Naive approaches that classify on full histories may still miss multi-turn jailbreaks, and cost scales poorly with number of turns.

## What is the core method / protocol?

- Use an M2S (Multi-turn to Single-turn) compression template to convert an entire multi-turn dialogue into one single prompt-like string.
- Fine-tune existing guardrail model families on these compressed strings (instead of full dialogues).
- Study multiple deterministic compression templates:
  - **hyphenize**, **numberize**, **pythonize** (formatting variants of the same underlying transcript information).
- Provide a complexity argument that training/inference with M2S reduces cost vs processing all turns directly.

## What are the key metrics?

- Attack detection **recall** on **SafeDialBench** (multi-turn jailbreak benchmark).
- Token counts / efficiency:
  - tokens per conversation at inference
  - total training tokens
- Comparisons across guardrail families: **LlamaGuard**, **Nemotron**, **Qwen3Guard**.

## What are the main results?

- Formal analysis: training cost reduced from **O(n^2)** to **O(n)** in number of turns *n* (as presented by the paper).
- On their training set (779 samples; avg 10.6 turns): **169K** tokens (M2S) vs **15.7M** tokens (multi-turn baseline) ≈ **93×** reduction.
- Best reported configuration: **Qwen3Guard + hyphenize**
  - **93.8%** attack detection recall
  - **94.6%** token reduction at inference (**3,231 → 173** tokens / conversation)
  - **+38.9 pp** recall improvement over baseline (paper claims baseline recall 54.9%).
- Strong model-template sensitivity (e.g., Nemotron best with numberize).
- Training on a single template can outperform mixing templates.

## How is this similar to GALILEO?

- If GALILEO involves multi-turn interaction traces (agent dialogues, tool-using episodes, or conversational logs), this is a concrete recipe for **compressing long histories into a single representation** while preserving the “safety-relevant” semantics.
- Emphasizes **efficiency + robustness tradeoffs** for long-context monitoring—likely aligned with GALILEO’s practical deployment constraints.

## How is this different from GALILEO?

- This is specifically about **guardrail classifiers for jailbreak detection**, not general reasoning, planning, or agentic task performance.
- Uses **deterministic formatting templates** as the main lever; no learned summarizer/compressor is required.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO uses principled representations or learned compression aligned with downstream objectives, it may generalize beyond “template engineering” and be less sensitive to format.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently requires full multi-turn context for scoring/classification, it may be paying a large token tax; this paper suggests a path to reduce cost drastically.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a related-work paragraph framing **M2S-style compression** as a pragmatic efficiency technique for long multi-turn traces (especially for safety monitoring / classification).
- [ ] Try a small ablation: compare any GALILEO classifier/scorer on (a) full trace vs (b) deterministic “flattened” trace (hyphen/number formats) to measure accuracy vs token savings.
- [ ] If GALILEO has to detect multi-step failures/attacks, test whether single-turn flattening preserves signal.

## Quotes / details to potentially cite

- “M2S requires only 169K tokens compared to 15.7M tokens for the multi-turn baseline — a 93× reduction.” (abstract)
- “Qwen3Guard with hyphenize compression, achieves 93.8% attack detection recall while reducing inference tokens by 94.6% (from 3,231 to 173 tokens per conversation).” (abstract)
