# Moral Susceptibility and Robustness under Persona Role-Play in Large Language Models

- Year: 2025
- Venue: arXiv
- Authors: Davi Bastos Costa, Felippe Alves, Renato Vicente
- URL: https://arxiv.org/abs/2511.08565
- BibTeX key (if we add it): costa2025moral-susceptibility-persona-roleplay
- Tags: persona, role-play, robustness, moral, evaluation, variability

## One-sentence takeaway

Persona role-play measurably shifts LLM moral-judgment outputs, and the paper proposes two variance-based metrics—moral robustness (within-persona stability) and moral susceptibility (across-persona sensitivity)—showing strong family effects for robustness and size trends for susceptibility.

## What problem does it solve?

- When LLMs are prompted to role-play different personas, their “moral judgments” (operationalized via MFQ ratings) can change; the paper aims to quantify (a) how stable a model is under repeated sampling for a fixed persona, and (b) how sensitive it is to changing personas.
- Provides a systematic, large-scale benchmarking protocol across many personas and models for persona-conditioned moral behavior.

## What is the core method / protocol?

- Instrument: Moral Foundations Questionnaire (MFQ), 30 items scored on an integer 0–5 scale; scores aggregated into five foundations (Harm/Care, Fairness/Reciprocity, In-group/Loyalty, Authority/Respect, Purity/Sanctity).
- Personas: 100 persona descriptions drawn from prior work (the paper cites Ge et al. 2025 “Scaling synthetic data creation”).
- Prompting: For each persona and each MFQ question, the model is instructed to role-play the persona and answer *one MFQ item at a time* (to reduce order effects). They instruct the model to start with a single integer rating (0–5), followed by reasoning.
- Repetition: Each persona–question pair is queried n=10 times to estimate within-persona mean/variance.
- Metrics (high-level):
  - **Moral robustness**: within-persona variability under repeated sampling (lower variance => more robust).
  - **Moral susceptibility**: across-persona variability (greater changes across personas => more susceptible).
  - They also discuss foundation-level decompositions and uncertainty estimates.

## What are the key metrics?

- Moral robustness (within-persona variability across repeated runs for fixed persona and question/foundation; aggregated).
- Moral susceptibility (across-persona variability; aggregated).
- Foundation-level MFQ profiles (no-persona baseline vs persona-averaged profiles).

## What are the main results?

- **Robustness**: model family explains most variance; model size shows *no systematic effect*.
- **Susceptibility**: mild family effect but a clearer within-family size trend; **larger variants tend to be more susceptible**.
- Reported ordering examples in the intro (needs verification from full tables): Claude family most robust (then Gemini, GPT-4), with others lower.
- Robustness and susceptibility are **positively correlated**, more strongly at the family level.

## How is this similar to GALILEO?

- Both study **behavioral instability under conversational/contextual “pressure”** (here: persona role-play; GALILEO: persona-based pressure in multi-turn with ground-truth).
- Both emphasize **trajectory/stability** rather than single-turn accuracy.
- Both benefit from separating sources of change: repeated sampling/decoding variance vs externally induced shifts.

## How is this different from GALILEO?

- Task differs: MFQ moral self-report ratings (subjective, no ground-truth) vs GALILEO’s **ground-truth tasks** where “flip to wrong” is well-defined.
- Interaction differs: this is essentially *single-item elicitation repeated* (many prompts), not a sustained multi-turn adversarial dialogue with a correct answer to maintain.
- Metrics are variance-based over personas/questions; GALILEO uses survival/TOF/recovery dynamics over turns under pressure + neutral re-asking control.

## Where GALILEO is stronger / cleaner (if true)

- Ground-truth framing allows clearer interpretation: “susceptibility” becomes “incorrect flip under pressure” rather than “moral profile shifts,” reducing normative ambiguity.
- Multi-turn protocol with explicit **neutral drift control** helps disentangle plain conversational drift from pressure-induced changes.
- Turn-level metrics (survival/TOF/recovery) are more directly connected to robustness in interactive settings.

## Where GALILEO is weaker / needs to improve

- GALILEO may not capture *persona-conditioned value shifts* as a first-class evaluation dimension; this paper offers an example of quantifying persona effects at scale.
- GALILEO should be careful to distinguish **decoding stochasticity** (within-condition variance) from **pressure effects**; this paper’s “robustness vs susceptibility” split is a useful conceptual lens.

## Action items for GALILEO (experiments / method / writing)

- [ ] In related work, cite as “persona role-play can systematically shift model judgments; variance decomposition into within-persona vs across-persona effects.”
- [ ] Consider adding a short paragraph in positioning: GALILEO focuses on *ground-truth* robustness under persona pressure; prior work uses subjective instruments (e.g., MFQ) without correctness.
- [ ] (Optional) Add an ablation or diagnostic: repeat identical turn under fixed persona/seed to estimate within-condition variance (analogous to “robustness”).

## Quotes / details to potentially cite

- Abstract (benchmark framing): “Using the Moral Foundations Questionnaire (MFQ), we introduce a benchmark that quantifies two properties: moral susceptibility and moral robustness, defined from the variability of MFQ scores across and within personas, respectively.”
- Method scale: “We evaluate |P|=100 persona descriptions … [and] |Q|=30 MFQ questions … [and] repeat each persona–question pair n=10 times …” (from the HTML version, Section 2.2).
