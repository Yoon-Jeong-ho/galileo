# Prompt-Reverse Inconsistency: LLM Self-Inconsistency Beyond Generative Randomness and Prompt Paraphrasing

- Year: 2025
- Venue: COLM 2025 (accepted)
- Authors: Jihyun Janice Ahn; Wenpeng Yin
- URL: https://arxiv.org/abs/2504.01282
- BibTeX key (if we add it): ahn2025prin
- Tags: self-inconsistency, llm-as-a-judge, evaluation, prompt-perturbation, logical-consistency

## One-sentence takeaway

LLMs can contradict themselves when judging the same answer set depending on whether they are asked for the *correct* vs *incorrect* options (Prompt-Reverse Inconsistency, PRIN), which is a direct threat to LLM-as-a-judge reliability but can be reduced with more explicit reasoning/negation scaffolding.

## What problem does it solve?

- Identifies and characterizes a *discriminative* inconsistency mode where an LLM’s judgments over a fixed candidate set are not logically complementary under prompt reversal ("Which are correct?" vs "Which are incorrect?").
- Motivates why common generative-inconsistency framings (sampling randomness; paraphrase sensitivity) are insufficient for evaluating LLMs used as graders/judges.

## What is the core method / protocol?

- Define PRIN on a multiple-choice(-like) setting: given a question + a pool of candidate answers, query the model with:
  - **Direct prompt**: ask for the correct answers.
  - **Reverse prompt**: ask for the incorrect answers.
  - Measure contradictions (e.g., options labeled correct in one query but also labeled incorrect in the reverse query).
- Evaluate across a mix of closed and open models (paper lists GPT-4/4o and several open LLMs) and across math/QA-style datasets (MATH, MathQA, EquationInference).
- Study: (i) prevalence across models/tasks; (ii) interaction with randomness & paraphrase perturbations; (iii) mitigations via added reasoning steps / explicit negation explanations; (iv) whether combining direct+reverse signals improves answer selection vs Self-Consistency majority vote.

## What are the key metrics?

- PRIN rate / inconsistency under prompt reversal (exact formalization varies; operationally: contradiction between direct-selection and reverse-selection outputs on the same candidate pool).
- Task accuracy when using different selection strategies (e.g., Self-Consistency voting vs direct/reverse-combined selection).

## What are the main results?

- PRIN is common across both closed and open models; some models that are relatively stable under sampling/paraphrase still show high PRIN (i.e., PRIN is not simply “more randomness”).
- PRIN does **not** positively correlate with randomness inconsistency or paraphrase inconsistency; it appears to be its own failure mode in judgment/logic consistency.
- Adding explicit reasoning paths before making the direct/reverse judgment, and giving additional explanatory scaffolding for *negation* in the reverse prompt, can reduce PRIN.
- Combining evidence from both direct and reverse prompting can outperform Self-Consistency on stronger models (the paper notes gains mainly for top models; weaker instruction-followers benefit less).

## How is this similar to GALILEO?

- If GALILEO uses any LLM-judge component (grading, filtering, preference labeling, verification), PRIN is a relevant threat model: judgments may depend on superficial prompt polarity rather than stable criteria.
- The paper’s mitigation theme—forcing an explicit reasoning chain + handling negation carefully—aligns with “make the judge explain itself” style robustness moves.

## How is this different from GALILEO?

- This work is primarily an *evaluation/analysis* paper about judge consistency; it does not propose a new task model or end-to-end system beyond selection strategies.
- Focuses on multiple-choice/candidate-pool judgment settings (math/QA), not necessarily the same domains/tasks as GALILEO.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO’s pipeline avoids relying on a single prompt-polarity judge (e.g., uses deterministic checks, explicit rubrics, or multi-view adjudication), it may be inherently less exposed to PRIN.

## Where GALILEO is weaker / needs to improve

- Any single-prompt judge decision (“is this correct?”) is vulnerable; PRIN suggests you should test invariance under prompt reversal (and more generally, under logically equivalent query transforms).

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a “prompt reversal” stress test for any LLM-as-a-judge step (direct vs reverse complement should match).
- [ ] If reversal fails, try mitigation: require short, structured reasoning + explicit handling of negation (e.g., “for each option, explain why it is incorrect”).
- [ ] In the paper related-work section, cite PRIN as distinct from sampling/paraphrase inconsistency, and as a caution for judge-based evaluation.

## Quotes / details to potentially cite

- Definition (paraphrased): PRIN occurs when, for the same question and candidate answers, the model gives conflicting judgments under “Which are correct answers?” vs “Which are incorrect answers?”.
- Motivation: PRIN undermines credibility of LLM-as-a-judge and indicates difficulty adhering to basic logical rules (complementarity under negation/reversal).
