# Shallow Robustness, Deep Vulnerabilities: Multi-Turn Evaluation of Medical LLMs

- Year: 2025
- Venue: NeurIPS 2025 Workshop on GenAI for Health (poster) (arXiv)
- Authors: Blazej Manczak; Eric Lin; Francisco Eiras; James O’Neill; Vaikkunth Mugunthan
- URL: https://arxiv.org/abs/2510.12255
- BibTeX key (if we add it): mancZAK2025medqaFollowup  
- Tags: multi-turn, robustness, medical, follow-up, context-manipulation, answer-stability

## One-sentence takeaway

Medical QA models that look strong in single-turn evaluation can collapse under *multi-turn* follow-up pressure—especially via **indirect context manipulations**—so “deep robustness” must be measured separately from “shallow robustness.”

## What problem does it solve?

- Existing medical LLM evaluation often assumes a one-shot question with a clean prompt; real consultations are iterative and include misleading context, authority pressure, and repeated re-questioning.
- Prior “robustness” work in medicine mainly targets *single-turn* prompt perturbations; this paper argues we need explicit multi-turn robustness measurement (and a taxonomy) to surface failures.

## What is the core method / protocol?

- Proposes **MedQA-Followup**: a framework + dataset for systematic *multi-turn* robustness evaluation on MedQA (USMLE-style MCQ).
- Introduces two axes:
  - **Shallow robustness**: perturbations in the *initial* prompt.
  - **Deep robustness**: perturbations in *follow-up turns after the model has already answered*.
- Adds a second axis:
  - **Indirect interventions** (subtle / not explicitly forcing a wrong answer):
    - *rethink* / neutral re-evaluation prompts that ask the model to reconsider.
    - *context manipulation*: additional context that implicitly supports an incorrect option or undermines the correct one.
    - (They also discuss *plausible wrong options* from prior work, but note it’s not naturally applicable as a follow-up intervention.)
  - **Direct interventions**: explicit attempts to push the model to a specific wrong answer (e.g., “wrong suggestion” framed with external justification / authority).
- Evaluates 5 state-of-the-art LLMs under controlled interventions; also studies **compounding** follow-ups (multiple interventions in sequence) to test degradation vs recovery.

## What are the key metrics?

- Accuracy under baseline vs interventions, reported separately for:
  - shallow (single-turn) vs deep (multi-turn follow-up) settings
  - indirect vs direct interventions
  - single follow-up vs **compounding** follow-ups (multiple turns)
- (From the framing) the key diagnostic is the *delta* from baseline accuracy as interventions accumulate.

## What are the main results?

- Models that remain “reasonably robust” to shallow perturbations can be extremely fragile under deep robustness tests.
- Reported headline failure: **Claude Sonnet 4** accuracy drops from **91.2% → 13.5%** under multi-turn context manipulation follow-ups.
- **Indirect, context-based interventions** can be *more damaging* than direct suggestions (counterintuitive, but important for realistic clinical settings where misinformation is often contextual).
- Under repeated/compounding interventions, models diverge: some degrade further, some partially recover, some even improve—so robustness is not monotonic and needs trajectory-aware reporting.

## How is this similar to GALILEO?

- Same core concern: **multi-turn interaction changes model behavior** and can surface robustness failures that one-shot eval misses.
- Emphasizes the need for structured, operator-based multi-turn protocols (follow-up operators; compounding sequences) rather than ad-hoc chatting.
- Highlights “pressure without new evidence” as a realistic deployment stressor, aligning with GALILEO’s drift/pressure framing.

## How is this different from GALILEO?

- Domain/task: medical MCQ (MedQA) rather than GALILEO’s broader “belief drift / pressure vs revision” positioning.
- Their primary outcome is **accuracy under follow-up interventions**; they do not foreground survival/time-to-event metrics (ToF, survival curves) or explicitly model drift-vs-evidence revision.
- Interventions are designed around clinical consultation realism (authority/context), not necessarily the full social-sycophancy space.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO includes explicit controls for **evidence-driven updates vs pressure-driven drift**, we can claim a cleaner causal separation than “follow-up vulnerability” alone.
- If GALILEO reports time-to-event / recovery metrics, we can provide more general trajectory diagnostics than raw accuracy deltas.

## Where GALILEO is weaker / needs to improve

- We likely need a clearer **taxonomy** presentation (like their shallow vs deep + indirect vs direct axes) to make our evaluation space legible.
- Their observation that *indirect* context framing can be more harmful suggests we should not over-focus on explicit “you are wrong” style adversaries.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add (or mirror) the **shallow vs deep robustness** terminology in related work; it’s a crisp, citeable framing.
- [ ] Ensure our operator set includes **context manipulation** follow-ups (indirect) in addition to explicit suggestion/rebuttal.
- [ ] Consider reporting a small “compounding interventions” slice (2+ sequential follow-ups) to show degradation vs recovery trends.

## Quotes / details to potentially cite

- “We introduce MedQA-Followup … distinguishing between **shallow robustness** … and **deep robustness** (maintaining accuracy when answers are challenged across turns).”
- “Counterintuitively, **indirect, context-based interventions** are often more harmful than direct suggestions …”
- “Accuracy dropping from **91.2%** to as low as **13.5%** for Claude Sonnet 4 …”
