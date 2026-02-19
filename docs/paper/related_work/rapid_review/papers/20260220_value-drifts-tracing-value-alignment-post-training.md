# Value Drifts: Tracing Value Alignment During LLM Post-Training

- Year: 2025
- Venue: arXiv
- Authors: Mehar Bhatia, Shravan Nayak, Gaurav Kamath, Marius Mosbach, Karolina Stanczak, Vered Shwartz, Siva Reddy
- URL: https://arxiv.org/abs/2510.26707
- BibTeX key (if we add it): bhatia2025valuedrifts
- Tags: value-alignment, post-training, drift, sft, rlhf, ppo, dpo, simpo, stance-eval

## One-sentence takeaway

Value-related stances in LLM outputs are largely set during SFT, and common preference-optimization stages often do not meaningfully re-shape these values unless the preference data has a clear value-gap and/or the algorithm induces different alignment dynamics.

## What problem does it solve?

- Prior value-alignment evaluations are mostly post-hoc (final checkpoint only), making it hard to attribute *when* and *why* a model started expressing certain values.
- This paper proposes a way to trace *value drift* over the course of post-training (SFT then preference optimization), disentangling dataset vs algorithm effects.

## What is the core method / protocol?

- Operationalize “values” via *stances* (support / neutral / oppose) on value-laden prompts for a topic.
- Build an evaluation set (V-PRISM) by filtering and clustering PRISM prompts, sampling 550 value-laden questions across 11 topical categories.
- For each checkpoint during post-training:
  - generate multiple responses per prompt (they mention 5 generations; temperature 0.7; max 256 tokens)
  - use an external LLM judge (they report GPT-4o) to classify stance and obtain stance probabilities
  - aggregate into topic-level stance distributions (value vectors)
- Define two drift diagnostics:
  - drift magnitude: how far the stance distribution moves over training
  - drift time: when in training most of that movement happens
- Controlled comparisons across:
  - model families / scales (Llama-3 and Qwen-3)
  - SFT datasets
  - preference optimization algorithms (PPO, DPO, SimPO)
  - a synthetic preference dataset where the value-gap can be manipulated.

## What are the key metrics?

- Topic-level stance distribution vectors (support/neutral/oppose probabilities).
- Drift magnitude and drift time (conceptual metrics summarizing change across checkpoints).
- Qualitative diagnosis of preference dataset “value-gap” (chosen vs rejected having similar stance distributions implies weak signal).

## What are the main results?

- SFT is typically the dominant driver of value alignment: values shift quickly during SFT toward the instruction-tuning data distribution.
- With standard preference datasets, preference optimization often makes little difference to values already established by SFT; the paper attributes this to chosen/rejected responses being too similar in value stance (low value-gap).
- With a synthetic preference dataset that has a controlled value-gap, preference optimization *can* reshape values, and outcomes depend on the algorithm (PPO vs DPO vs SimPO can lead to different value alignment even with the same preference data).

## How is this similar to GALILEO?

- GALILEO cares about drift/instability of model behavior under interaction; this work formalizes “drift” and measures it as a distributional change over time.
- Their decomposition of *where drift comes from* (stage, dataset, algorithm) is aligned with GALILEO’s goal of attributing behavior changes to specific training/interaction factors.

## How is this different from GALILEO?

- They study drift across *training checkpoints* in post-training, not multi-turn conversational pressure / repeated interactions.
- Their “values” are stance distributions on curated value-probing prompts; GALILEO’s focus is broader (multi-turn robustness, sycophancy/persuasion, belief revision vs drift controls).
- Heavy reliance on an external LLM judge for stance scoring; GALILEO may want judge-robustness and multi-judge agreement tests.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO evaluates robustness across repeated dialogue rounds and adversarial pressure, it can capture interaction-specific failure modes that checkpoint-based post-training tracing does not.
- GALILEO can emphasize causal interventions in the *interaction loop* (prompting, memory, persona pressure), not just training stages.

## Where GALILEO is weaker / needs to improve

- If GALILEO does not currently report “when drift happens” as a time-localized metric, this paper suggests a clear way to summarize dynamics (drift time) rather than only endpoints.
- If GALILEO does not explicitly test preference-data value-gap (chosen vs rejected stance separation), it may miss a key explanation for why preference optimization does or does not change target behaviors.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add drift diagnostics analogous to “magnitude” and “time” for multi-turn settings (e.g., when during a conversation a model’s stance/belief shifts the most).
- [ ] When analyzing any preference-optimization component, quantify a “behavior-gap” between preferred vs dispreferred responses for GALILEO-relevant traits (sycophancy, compliance under pressure, belief revision quality).
- [ ] Consider a synthetic preference dataset where the target trait gap is explicitly controlled, to separate dataset signal from algorithm effects.
- [ ] Run judge-sensitivity checks: replicate stance/drift estimates with at least one alternative judge model or a small ensemble.

## Quotes / details to potentially cite

- The paper’s central framing: “value drifts” are shifts in a model’s expressed stances over the course of post-training, and tracing them can attribute values to stages/datasets/algorithms.
- Key qualitative claim: SFT generally establishes values; preference optimization on standard datasets “rarely re-aligns” these values due to low contrast between chosen and rejected responses.
- Evaluation artifact: V-PRISM derived from PRISM; 550 value-laden questions across 11 topical categories; stance labels support/neutral/oppose inferred via GPT-4o.
