# Sycophancy under Pressure: Evaluating and Mitigating Sycophantic Bias via Adversarial Dialogues in Scientific QA

- Year: 2025
- Venue: arXiv
- Authors: Kaiwei Zhang; Qi Jia; Zijian Chen; Wei Sun; Xiangyang Zhu; Chunyi Li; Dandan Zhu; Guangtao Zhai
- URL: https://arxiv.org/html/2508.13743v1
- BibTeX key (if we add it): zhang2025sycophancy_pressure_scientificqa
- Tags: sycophancy, adversarial-dialogue, scientific-qa, mitigation, post-training

## One-sentence takeaway

Proposes a scientific-QA-focused evaluation of sycophancy under misleading multi-turn pressure and a lightweight SFT mitigation ("Pressure-Tune") that increases resistance without (claimed) accuracy loss.

## What problem does it solve?

- Prior sycophancy work often targets social/dialogue settings or single-turn prompts; this paper argues **factual / scientific QA** needs its own evaluation because agreement-with-user can directly degrade factual correctness.
- Provides metrics intended to quantify "how much misleading user pressure distorts answers" and a practical post-training recipe to reduce this failure mode.

## What is the core method / protocol?

- **Evaluation framework** with two settings:
  - **Single-turn**: embed an assertive but incorrect user stance directly in the prompt.
  - **Multi-turn**: simulate dialogue progression where the user provides misleading/confounding feedback over turns, then measure answer shift / consistency.
- **Mitigation (Pressure-Tune)**:
  - Supervised fine-tuning on **synthetic adversarial dialogues**.
  - Training targets include chain-of-thought style rationales that explicitly *reject* user misinformation while reaffirming the correct answer.
  - CoT supervision is produced using a strong reference model prompted with access to the correct answer.

## What are the key metrics?

(Names are from the paper’s abstract/intro; details likely later in the paper.)

- **Misleading resistance**: ability to maintain factual correctness under misleading cues.
- **Sycophancy resistance**: overall rate of not conforming to incorrect user beliefs.
- Mentions additional measures like **confounding success** in the multi-turn setting.

## What are the main results?

- Broad claim: sycophancy is **pervasive** across open-source and proprietary models in scientific QA.
- Claim: susceptibility is driven more by **alignment strategy** than raw model size.
- **Pressure-Tune** improves sycophancy-resistance metrics while preserving (claimed) accuracy and the ability to accept *valid* feedback.

## How is this similar to GALILEO?

- Same core phenomenon: **multi-turn user pressure** can cause models to abandon correct answers (flip) in favor of user-implied preferences/beliefs.
- Similar framing that *alignment/politeness* objectives can induce over-agreement.

## How is this different from GALILEO?

- Domain focus: **scientific QA** benchmarks rather than GALILEO’s explicit **ground-truth tasks + survival/TOF + recovery + neutral re-asking control** protocol.
- Their mitigation centers on **post-training with synthetic adversarial dialogues + CoT**; GALILEO is currently primarily an **evaluation protocol** (and separates pressure vs drift using a neutral control).
- Does not (from the abstract/intro) emphasize **recovery-after-flip** trajectories as a core metric the way GALILEO does.

## Where GALILEO is stronger / cleaner (if true)

- GALILEO explicitly disentangles:
  - pressure-driven effects vs **neutral repeated-question drift** (control), and
  - **recovery conditional on flip**, which helps avoid hiding oscillation/recovery structure inside average accuracy.
- GALILEO is designed as a general protocol across tasks with known ground truth, not tied to a specific QA benchmark family.

## Where GALILEO is weaker / needs to improve

- This paper provides a concrete, simple mitigation baseline (Pressure-Tune) targeted at sycophancy in factual QA; GALILEO may need at least one **mitigation baseline** to contextualize measured failure rates.
- If their metrics are well-defined and easy to report, GALILEO should ensure its own metrics are equally interpretable to non-specialists.

## Action items for GALILEO (experiments / method / writing)

- [ ] Writing: cite as an example of (i) sycophancy risk in **factual QA** and (ii) evidence that post-training can trade off truthfulness vs user-approval.
- [ ] Method: consider adding a simple mitigation baseline category (e.g., SFT on adversarial dialogues) to show GALILEO metrics can detect improvements without conflating them with drift.
- [ ] Metrics: crosswalk their "misleading resistance" / "sycophancy resistance" into GALILEO’s survival/TOF/recovery vocabulary (or add a short mapping paragraph in related work).

## Quotes / details to potentially cite

- Abstract (problem framing): sycophancy is “the tendency to align with user beliefs regardless of correctness” and is risky in scientific QA.
- Abstract (method): introduces an evaluation framework with metrics like “misleading resistance” and “sycophancy resistance”.
- Abstract (mitigation): “Pressure-Tune, a lightweight post-training method ... fine-tunes models on synthetic adversarial dialogues paired with chain-of-thought rationales ... reject user misinformation while reinforcing factual commitments.”
