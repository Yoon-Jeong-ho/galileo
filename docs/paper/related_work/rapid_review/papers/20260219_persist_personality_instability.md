# Persistent Instability in LLM's Personality Measurements: Effects of Scale, Reasoning, and Conversation History

- Year: 2026 (arXiv 2025)
- Venue: AAAI 2026 (AI Alignment track); arXiv:2508.04826v3
- Authors: Tommaso Tosato, Saskia Helbling, Yorguin-Jose Mantilla-Ramos, Mahmood Hegazy, Alberto Tosato, David John Lemay, Irina Rish, Guillaume Dumas
- URL: https://arxiv.org/abs/2508.04826
- BibTeX key (if we add it): tosato2026persist
- Tags: personality, behavioral-stability, prompt-sensitivity, reasoning, conversation-history, evaluation

## One-sentence takeaway

Across 25 open-source LLMs and 2M+ questionnaire responses, personality “self-report” measurements are highly unstable under minor prompt/ordering/history/reasoning changes, and scaling/reasoning/history do not reliably improve stability.

## What problem does it solve?

- Quantifies how *stable* (or not) LLM personality measurements are under realistic deployment perturbations (question order, paraphrase, persona prompt, reasoning mode, conversation history), which matters for safety certification and any “personality-based” alignment or monitoring.

## What is the core method / protocol?

- PERSIST framework: run psychometric-style personality questionnaires against many LLMs repeatedly while varying:
  - model scale (1B–685B; 25 open-source models)
  - question permutations (reported 250 reorderings)
  - paraphrasing settings (reported 100)
  - persona prompts (baseline helpful assistant + detailed personas, incl. misaligned)
  - reasoning vs non-reasoning modes (chain-of-thought style)
  - conversation history modality (carry prior turns vs not)
- Instruments:
  - standard human psychometrics: Big Five Inventory (BFI), Short Dark Triad (SD3)
  - “LLM-adapted” personality questionnaires (intended to be more ecologically valid for LLMs)
- Output: distributions / variability of trait scores across repeats / conditions; compare stability across interventions.

## What are the key metrics?

- Variability / instability of measured trait scores across runs and perturbations, summarized primarily via standard deviation (SD) on 5-point scales (plus comparative analyses across conditions).

## What are the main results?

- Question reordering alone can cause large shifts in measured personality traits.
- Scaling helps only modestly: even 400B+ models reportedly still show SD > 0.3 on 5-point scales.
- Reasoning can *increase* instability (different justifications → different answers).
- Conversation history can exacerbate variability, especially for smaller models.
- Detailed persona prompting has mixed effects; misaligned personas tend to increase variability relative to a helpful assistant baseline.
- LLM-adapted questionnaires are not materially more stable than the original human-centric ones.

## How is this similar to GALILEO?

- Motivates and operationalizes “behavioral consistency under perturbations” as a first-class evaluation target, aligned with safety/reliability framing.
- Highlights that *prompt-level* and *interaction-level* variation (order, paraphrase, history) can dominate measured behavior—relevant for any GALILEO evaluation protocol meant to be robust and reproducible.

## How is this different from GALILEO?

- Focuses on *personality questionnaire measurements* (self-report style) rather than task/agent performance, capability, or application-grounded safety metrics.
- Emphasizes measurement instability (psychometrics + prompt sensitivity) more than intervention design for improving stability.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO is grounded in application tasks / behavioral tests beyond self-report questionnaires, GALILEO can avoid some construct-validity pitfalls of “personality score” proxies.
- GALILEO can explicitly design evaluation to separate *model stochasticity* from *prompt sensitivity* (e.g., fixed decoding seeds/temperatures and controlled perturbation suites).

## Where GALILEO is weaker / needs to improve

- If GALILEO currently assumes stable “traits”/profiles or uses single-shot measurements, this paper suggests that needs rethinking: stability must be measured, not assumed.
- If GALILEO uses reasoning-mode comparisons, it should check whether enabling reasoning changes variance as well as mean behavior.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a small “stability under perturbations” subsection: for any behavioral attribute GALILEO measures, report variability across (a) question/order, (b) paraphrase, (c) history/no-history, (d) reasoning/no-reasoning.
- [ ] When reporting improvements, include *variance* metrics (e.g., SD across perturbations) in addition to means.
- [ ] If GALILEO uses persona prompts, explicitly test misaligned/edge personas and quantify whether they increase instability.

## Quotes / details to potentially cite

- “Question reordering alone can introduce large shifts in personality measurements.”
- “Scaling provides limited stability gains: even 400B+ models exhibit standard deviations > 0.3 on 5-point scales.”
- “Interventions expected to stabilize behavior, such as reasoning and inclusion of conversation history, can paradoxically increase variability.”
