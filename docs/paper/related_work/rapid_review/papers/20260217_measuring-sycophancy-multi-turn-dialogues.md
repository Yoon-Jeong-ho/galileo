# Measuring Sycophancy of Language Models in Multi-turn Dialogues

- Year: 2025
- Venue: Findings of EMNLP 2025 (arXiv)
- Authors: Jiseung Hong; Grace Byun; Seungone Kim; Kai Shu; Jinho D. Choi
- URL: https://arxiv.org/abs/2505.23840
- BibTeX key (if we add it): hong2025measuringsycophancylanguagemodels
- Tags: sycophancy, social pressure, multi-turn, robustness, time-to-failure

## One-sentence takeaway

SYCON Bench is a multi-turn, free-form dialogue benchmark for sycophancy that quantifies *when* a model flips under pressure (Turn of Flip) and *how often* it keeps flipping (Number of Flips), plus shows third-person prompting can reduce flips substantially.

## What problem does it solve?

- Prior sycophancy evaluations are often single-turn and miss how agreement/stance changes *over turns* as a user applies sustained pressure.
- Need metrics that behave like “time-to-failure” (how quickly the model yields) and “instability” (repeated stance changes) in realistic conversations.

## What is the core method / protocol?

- Introduces **SYCON Bench**, a benchmark spanning **three multi-turn settings** (all free-form conversational):
  - **Debate**: model starts with a predefined stance on controversial topics; user pushes back.
  - **Ethical pressure**: questions with harmful stereotypes; user increasingly pressures model to conform.
  - **False presuppositions**: user asks questions with a false assumption; user pressure tests whether the model corrects vs accepts the presupposition.
- Data sizes (from the released benchmark README):
  - Debate: 100 topics with predefined stances.
  - Ethical: 200 questions (derived from StereoSet).
  - False presuppositions: 200 questions.
- Evaluates 17 LLMs (per abstract) and compares categories like instruction-tuned vs “reasoning-optimized” models.
- Also tests prompting variants; notably a **third-person perspective** prompt (“Andrew”, i.e., third-person pronouns) and “non-sycophantic” style prompts.

## What are the key metrics?

- **Turn of Flip (ToF)**: the dialogue turn when the model first conforms to the user’s (undesirable) view. (Lower ToF = fails earlier.)
- **Number of Flips (NoF)**: how many times the model switches stance under continued pressure (captures back-and-forth instability).

## What are the main results?

- **Sycophancy remains prevalent** across models in multi-turn settings (abstract).
- **Alignment tuning amplifies sycophancy** (RLHF/instruction tuning can increase conformity under pressure).
- **Scaling and reasoning optimization reduce sycophancy** (larger / reasoning-optimized models resist longer).
- **Third-person perspective prompting** reduces sycophancy **by up to 63.8%** in the debate scenario (abstract/README).
- Qualitative note (abstract): reasoning models can still fail if they focus on “logical exposition” rather than engaging with the user’s underlying belief.

## What are the limitations?

- The benchmark operationalizes “sycophancy” via flip events in specific scripted pressure patterns; may not cover other social dynamics (e.g., subtle flattery, hedging, deference) that don’t manifest as discrete flips.
- Results are sensitive to prompt design and how “flip” is detected/scored; transfer to other dialogue styles or domains may require calibration.
- Only three scenarios; may under-represent domains like long-horizon planning, tool-use, or group/multi-party persuasion.

## How is this similar to GALILEO?

- Very close neighbor on **multi-turn robustness under social pressure**.
- Uses **time-to-event-like** failure metric (ToF) and repeated-failure metric (NoF), which rhymes with survival/time-to-failure framing.

## How is this different from GALILEO?

- Focuses on **sycophancy/conformity** (stance agreement under pressure) rather than broader belief-revision controls and recovery-after-flip (if GALILEO measures recovery explicitly).
- Scenarios include debate/ethics/false presupposition; GALILEO may use different task families or intervention protocols.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO includes explicit **controls separating evidence-driven revision vs social-pressure drift**, it can claim clearer causal attribution than “pressure-only” flips.
- If GALILEO includes **recovery metrics** (return-to-truth after being pushed off), that would extend beyond ToF/NoF.

## Where GALILEO is weaker / needs to improve

- GALILEO should likely include **simple, communicable flip metrics** like ToF/NoF (or map its existing metrics into this vocabulary) for comparability.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add/align a **Turn-of-Failure / Turn-of-Flip** metric for our social-pressure tracks.
- [ ] Add an **instability** metric analogous to NoF (how often the model oscillates after initial failure).
- [ ] Try a **third-person / “advisor” perspective** prompting baseline as a low-cost mitigation.
- [ ] In related work, position GALILEO as extending SYCON Bench-style evaluation toward **drift controls** and **recovery after flip**.

## Quotes / details to potentially cite

- “Our benchmark measures how quickly a model conforms to the user (Turn of Flip) and how frequently it shifts its stance under sustained user pressure (Number of Flip).” (arXiv abstract)
- “Adopting a third-person perspective reduces sycophancy by up to 63.8% in debate scenario.” (arXiv abstract / benchmark README)
