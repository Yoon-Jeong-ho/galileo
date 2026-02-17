# RobustExplain: Evaluating Robustness of LLM-Based Explanation Agents for Recommendation

- Year: 2026
- Venue: Companion Proceedings of the ACM Web Conference 2026 (WWW Companion ’26)
- Authors: Guilin Zhang; Kai Zhao; Jeffrey Friedman; Xu Chu
- URL: https://arxiv.org/abs/2601.19120
- BibTeX key (if we add it): RobustExplainZhang2026
- Tags: robustness, multi-turn (history), perturbations, explanation-stability, recommendation

## One-sentence takeaway

RobustExplain proposes a perturbation-based evaluation suite plus multi-dimensional “stability” metrics to quantify how much LLM-generated recommendation explanations change when user-history inputs are realistically noisy or drifting.

## What problem does it solve?

- LLMs are increasingly used as *explanation agents* in recommender systems, but prior evaluation focuses on fluency/relevance on clean/static inputs.
- In practice, user histories are noisy (accidental clicks), temporally inconsistent, incomplete (missing metadata), and non-stationary (preference drift), so explanations may become unstable and reduce user trust.

## What is the core method / protocol?

- Define robustness as expected similarity between an explanation generated from an original user interaction history and the explanation generated from a *perturbed* version of that history, holding the recommended item fixed.
- Introduce a perturbation taxonomy with **5 perturbation types**, each evaluated at **5 severity levels (1–5)**:
  - **Noise injection:** add random interactions (Level 5 ≈ +50% interactions).
  - **Temporal shuffle:** permute interaction order (Level 5 shuffles all).
  - **Behavior dilution:** inject interactions from least-engaged categories (shared account / multi-user device).
  - **Category drift:** replace a fraction of interactions with other-category items (Level 5 up to ≈ 50% replaced).
  - **Missing values:** remove ratings/timestamps/category fields (Level 5 ≈ 50% metadata missing).
- Run multiple LLM explanation generators (authors report 4 models spanning ~7B–70B parameters) to produce (original, perturbed) explanation pairs across perturbations/severities.

## What are the key metrics?

A **multi-dimensional robustness** score built from four complementary consistency measures between original explanation *e* and perturbed explanation *e′*:

- **Semantic consistency:** bag-of-words cosine similarity (TF vectors) as an overall meaning-preservation proxy.
- **Keyword stability:** overlap/consistency of salient keywords (intended to track whether the “reasons” change).
- **Structural preservation:** whether the explanation keeps similar structure (e.g., formatting / ordering / sentence-level organization).
- **Length variation:** stability of explanation length under perturbation.

(Exact aggregation details are described in the paper; the key point is the decomposition into semantic/keyword/structure/length stability to make failures interpretable.)

## What are the main results?

- Current LLM explanation agents show **only moderate robustness** under realistic user-history perturbations.
- Reported average consistency scores are around **~0.50** (substantial sensitivity).
- **Model size helps**: larger models achieve up to **~8% higher stability** than smaller ones.
- Different perturbation types expose different weaknesses (e.g., robustness to random noise vs robustness to drift/metadata loss is not uniform).

## How is this similar to GALILEO?

- Same general philosophy: evaluate *trajectory/interaction robustness* under controlled perturbations rather than only single-shot quality.
- Explicitly distinguishes perturbation families that resemble GALILEO-relevant stressors:
  - “noise injection” ↔ irrelevant distractors
  - “category drift” ↔ distribution shift / preference drift
  - “missing values” ↔ partial observability / incomplete state
- Uses a *stability* framing (similarity between baseline and perturbed outputs) that can complement time-to-failure style metrics.

## How is this different from GALILEO?

- Domain is **recommendation explanations**, not social pressure / sycophancy / persuasion.
- Perturbations are on **user-history records**, not adversarial dialogue acts, social manipulation, or multi-agent pressure.
- Metrics are mostly similarity-based (semantic/keyword/structure/length), not belief-state tracking, flip rates, recovery, or survival/time-to-event.

## Where GALILEO is stronger / cleaner (if true)

- GALILEO’s target setting (multi-turn social pressure, belief revision vs drift controls, recovery after flip) demands *causal* distinctions that similarity metrics may not capture.
- If GALILEO includes explicit “ground-truth belief” labels and turn-level event annotations (flip/recover), it can provide sharper failure semantics than generic text similarity.

## Where GALILEO is weaker / needs to improve

- RobustExplain’s **taxonomy + severity ladder** is a clean, reproducible way to stress-test systems; GALILEO could benefit from a similarly explicit perturbation grid (even if the perturbations are social).
- The multi-dimensional decomposition (semantic vs keyword vs structure vs length) is a nice “debuggable” reporting format.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a “perturbation taxonomy + severity levels” table for GALILEO’s stressors (e.g., distractor injection, social-pressure escalation, evidence vs assertion, memory corruption/omission), mirroring RobustExplain’s clarity.
- [ ] Consider reporting a *multi-dimensional stability panel* alongside flip/recovery metrics (e.g., semantic stability vs stance stability vs citation/evidence stability).
- [ ] In related work, cite RobustExplain as an example of *task-level robustness evaluation for LLM agents* under realistic input noise, and contrast with GALILEO’s social-pressure / belief-dynamics focus.

## Quotes / details to potentially cite

- “RobustExplain introduces five realistic user behavior perturbations … and a multi-dimensional robustness metric capturing semantic, keyword, structural, and length consistency.”
- “Experiments … (7B–70B) show that current models exhibit only moderate robustness, with larger models achieving up to 8% higher stability.”
