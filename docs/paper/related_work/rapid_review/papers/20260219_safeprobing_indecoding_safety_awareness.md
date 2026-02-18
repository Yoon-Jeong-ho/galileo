# Defending Large Language Models Against Jailbreak Attacks via In-Decoding Safety-Awareness Probing

- Year: 2026
- Venue: arXiv
- Authors: Yinzhi Zhao, Ming Wang, Shi Feng, Xiaocui Yang, Daling Wang, Yifei Zhang
- URL: https://arxiv.org/abs/2601.10543v1
- BibTeX key (if we add it): zhao2026safeprobing
- Tags: jailbreak, defense, decoding-time detection, self-monitoring, safety

## One-sentence takeaway

Probe a model’s *latent* safety awareness during decoding by appending a neutral disclaimer prefix and measuring likelihood of an “illegal and unethical” continuation, then interrupt generation early when the signal triggers.

## What problem does it solve?

- Safety-aligned LLMs can be jailbroken; existing defenses (decoding constraints or post-hoc detectors) can be brittle, slow, or harm utility.
- Goal: detect harmful generation *as it emerges* with low over-refusal and minimal quality degradation.

## What is the core method / protocol?

- Key observation: even when jailbroken, models sometimes add a disclaimer after producing harmful content, suggesting an internal safety-related signal exists but is overridden by “continue fluently” pressures.
- Define a “virtual disclaimer” probe string: "Note that this is illegal and unethical".
  - During probing, append the prefix "Note that this is" and compute token-level loss / likelihood for the continuation "illegal and unethical".
  - Intuition: if the model recognizes the prior content as harmful, it assigns higher probability (lower loss) to the moralizing continuation; if benign, the continuation is incoherent (higher loss).
- In-decoding probing:
  - Randomly sample checkpoints during generation (paper uses 20% of decoding steps).
  - At each checkpoint, compute the probe loss on the partial generation.
  - If loss < threshold, halt generation and return a fixed refusal.
  - Generation itself is still produced by the base model; probing can be done with a safety-enhanced copy.
- Optional lightweight fine-tuning (“enhance safety-awareness”):
  - Use a small labeled set (SafeRLHF) to train the model to agree with the appended disclaimer on harmful outputs and disagree on benign outputs.
  - Uses a bounded mapping of disclaimer loss and an MSE objective, plus instruction-following regularization (UltraFeedback) to reduce utility loss.

## What are the key metrics?

- Defense Success Rate (DSR) against jailbreak attacks (block harmful outputs).
- Over-refusal rate on benign prompts.
- Utility / response quality retention (paper mentions math ability and general utility benchmarks).
- Runtime overhead (extra forward passes for probing at checkpoints).

## What are the main results?

- In-decoding probing provides clearer separation between harmful vs benign than only probing after full completion (“last-check”), because the signal is strongest immediately when harmful content appears.
- Reported to improve jailbreak defense across multiple attacks and models (Qwen2.5-7B-Instruct, Mistral-7B-Instruct-v0.3; additional models in appendix) while keeping over-refusal low and preserving response quality.

## How is this similar to GALILEO?

- Same broad direction: using *during-generation* signals rather than only pre-generation filtering or post-hoc classification.
- Emphasizes minimal intervention: keep base decoding largely unchanged unless a detector triggers.

## How is this different from GALILEO?

- Uses an explicit, hand-crafted textual probe (“Note that this is … illegal and unethical”) and measures likelihood/loss of a specific continuation as the safety signal.
- Primarily a detection-and-refusal mechanism (halt + fixed refusal) rather than producing a “safe alternative answer” by construction.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO provides a more general, content-agnostic safety signal (not tied to one disclaimer phrase), it may generalize better across topics/languages.
- If GALILEO yields controllable safe completion rather than hard refusal, it may preserve more utility in borderline cases.

## Where GALILEO is weaker / needs to improve

- If GALILEO lacks an explicit decoding-time monitor, SafeProbing suggests a simple add-on that may catch attacks that bypass prompt-level defenses.
- If GALILEO’s detector is purely post-hoc, this paper argues timing matters (signal decays after generation completes).

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a decoding-time “probe head” baseline: at random checkpoints, append a short neutral prefix and measure likelihood of a safety clause; compare last-check vs in-decoding.
- [ ] Evaluate the trade-off curve: checkpoint sampling rate (e.g., 5%, 10%, 20%) vs overhead vs detection.
- [ ] Test sensitivity to the chosen probe phrase (alternate wordings; multilingual; different moral clauses).
- [ ] If writing related work, position this as “in-decoding self-monitoring via likelihood probing” vs classic output classifiers.

## Quotes / details to potentially cite

- Observation: models sometimes append disclaimers after generating harmful content, implying latent safety awareness during decoding.
- Core probe: append "Note that this is" and measure probability / loss of continuing with "illegal and unethical" as harmfulness indicator.
- Method detail: sample checkpoints during decoding (paper uses 20% of steps) and interrupt generation if any checkpoint’s probe loss crosses threshold.
