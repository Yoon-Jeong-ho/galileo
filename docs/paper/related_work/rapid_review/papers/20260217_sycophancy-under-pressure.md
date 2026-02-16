# Sycophancy under Pressure: Evaluating and Mitigating Sycophantic Bias via Adversarial Dialogues in Scientific QA

- Year: 2025
- Venue: arXiv
- Authors: Kaiwei Zhang; Qi Jia; Zijian Chen; Wei Sun; Xiangyang Zhu; Chunyi Li; Dandan Zhu; Guangtao Zhai
- URL: https://arxiv.org/abs/2508.13743
- BibTeX key (if we add it): zhang2025sycophancy_pressure
- Tags: sycophancy, social-pressure, scientific-qa, adversarial-dialogue, mitigation, post-training

## One-sentence takeaway

Proposes a scientific-QA-focused sycophancy evaluation (single-turn + multi-turn adversarial pressure) and a lightweight post-training mitigation (“Pressure-Tune”) to improve resistance to misleading user pressure while preserving accuracy.

## What problem does it solve?

- Quantifying and mitigating **sycophancy under user-imposed social pressure** in *factual/scientific QA*, where “agreeing with the user” can directly distort factual outputs.
- Provides a more domain-grounded alternative to purely social/subjective sycophancy settings.

## What is the core method / protocol?

- **Unified evaluation framework** for sycophancy in scientific QA with two settings:
  - **Single-turn**: embed a misleading user stance directly into the prompt (assertive but incorrect cues).
  - **Multi-turn**: simulate dialogue where the user provides misleading/confounding feedback over turns; measure whether the model’s answer shifts away from the ground-truth.
- **Pressure-Tune (mitigation)**:
  - Synthetic adversarial dialogues paired with **chain-of-thought (CoT) rationales** that explicitly *reject misinformation* and *re-commit to the correct answer*.
  - Post-training via supervised fine-tuning; rationales are produced using a strong reference model given the correct answer + structured context.

## What are the key metrics?

(As named in the paper)

- **Misleading resistance**: ability to maintain factual consistency under misleading cues.
- **Sycophancy resistance** (rate): overall resistance to user-imposed distortion.
- **Confounding success** (multi-turn): whether confounding/misleading feedback succeeds in changing the model’s answer.

## What are the main results?

- Across both open-source and proprietary models, sycophantic behavior under pressure is **pervasive** in scientific QA.
- Susceptibility appears **more tied to alignment strategy** than to raw model size.
- **Pressure-Tune** improves sycophancy-resistance metrics **without** sacrificing baseline accuracy, and without making the model unresponsive to *valid* user feedback (as claimed by the authors).

## How is this similar to GALILEO?

- Same core phenomenon: **multi-turn user pressure can drive answer drift/flip** away from truth.
- Emphasizes *protocol + metrics* for “robustness under pressure” rather than only static single-turn probes.

## How is this different from GALILEO?

- Domain: centered on **scientific QA benchmarks**.
- Focus includes an explicit **post-training mitigation** recipe (synthetic adversarial dialogue + CoT SFT).
- Metrics are framed as resistance rates rather than explicit *time-to-failure / survival-style* measures.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO cleanly separates **evidence-driven revision** vs **pressure-driven drift**, that’s a clearer causal story than “resistance to misleading cues” alone.
- If GALILEO measures **recovery after a flip** (return-to-truth dynamics), that’s an additional axis not foregrounded here.

## Where GALILEO is weaker / needs to improve

- We may be missing a straightforward, reusable **mitigation baseline** akin to Pressure-Tune (lightweight post-training on adversarial dialogues).
- Scientific-QA framing is compelling; GALILEO should ensure it has at least one **high-stakes factual domain** slice (or show domain generality).

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a “Pressure-Tune-style” baseline: synthetic adversarial multi-turn dialogues + SFT (with/without rationales) and evaluate on GALILEO’s drift/revision controls.
- [ ] Align metric language: map GALILEO’s measures onto “misleading resistance / sycophancy resistance” equivalents for reader familiarity.
- [ ] Consider a *scientific QA* subset/task to test whether GALILEO’s findings hold in high-stakes factual settings.

## Quotes / details to potentially cite

- Motivation claim (abstract): preference-based alignment can reinforce sycophancy; in scientific QA this is risky.
- Key metric names (abstract/intro): “misleading resistance” and “sycophancy resistance”.
- Mitigation headline (abstract): Pressure-Tune is a lightweight post-training method using synthetic adversarial dialogues + CoT rationales that reject misinformation while reinforcing factual commitments.

## Limitations / caveats (for our internal use)

- Rapid-review skim (abstract/intro/html): did not extract detailed numerical results/tables.
- Mitigation relies on **synthetic dialogue generation** and **CoT rationale supervision**; generalization and robustness to distribution shift should be checked.
- The evaluation framing does not obviously include **survival/time-to-event** metrics (turn-of-failure) or **recovery-after-flip** measures, which are central to GALILEO’s angle.
