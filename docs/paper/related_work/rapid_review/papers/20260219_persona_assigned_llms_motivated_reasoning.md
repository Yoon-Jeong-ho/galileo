# Persona-Assigned Large Language Models Exhibit Human-Like Motivated Reasoning

- Year: 2025
- Venue: arXiv
- Authors: Saloni Dash; Amélie Reymond; Emma S. Spiro; Aylin Caliskan
- URL: https://arxiv.org/abs/2506.20020
- BibTeX key (if we add it): dash2025persona_assigned_motivated_reasoning
- Tags: persona, motivated-reasoning, veracity, evidence

## One-sentence takeaway

Assigning political/sociodemographic personas to LLMs measurably shifts their reasoning toward identity-congruent conclusions (hurting misinformation discernment and skewing evidence interpretation), and common prompt-only “debiasing” is largely ineffective.

## What problem does it solve?

- Quantifies whether persona prompting induces *motivated reasoning*-like behavior (identity-protective, selectively biased reasoning) in LLMs.
- Tests risks of persona/personification features for factual judgment on socially/politically charged topics.

## What is the core method / protocol?

- Persona assignment: 8 personas spanning 4 political + sociodemographic attributes (paper framing: “political and socio-demographic attributes”).
- Models: 8 LLMs (mix of proprietary + open-source; paper mentions “4 OpenAI models, and 4 open source models”).
- Tasks (borrowed from human motivated-reasoning / cog-psych paradigms):
  - Headline veracity discernment: classify real vs synthetic/misinformation headlines.
  - Scientific evidence evaluation: interpret numeric evidence about a contested policy topic (example shown: gun control → crime), where the “ground truth” can be congruent or incongruent with the induced persona.
- Analysis focus: performance differences with vs without personas; congruent vs incongruent conditions; regression-style predictors comparing “motivated reasoning” vs “analytic reasoning”.
- Mitigation attempt: prompt-based debiasing (explicitly includes chain-of-thought prompting; other debiasing prompt strategies per §4.3).

## What are the key metrics?

- Accuracy / veracity discernment performance (headline task).
- Evidence evaluation correctness (numeric scientific evidence task), stratified by whether ground truth is congruent vs incongruent with political persona.
- Effect sizes described as relative reductions (e.g., “up to 9% reduced veracity discernment”) and likelihood/odds shifts (“up to 90% more likely” in one congruent condition).

## What are the main results?

- Persona assignment can *decrease* veracity discernment:
  - “High School educated” persona yields up to ~9% reduced veracity discernment vs no-persona baseline.
- Political personas induce identity-congruent evidence interpretation:
  - Up to ~90% more likely to correctly evaluate evidence on gun control *when the ground truth aligns* with the induced political identity, with reduced performance when evidence conflicts with that identity.
- In the headline task, motivated-reasoning signal is a statistically significant predictor of veracity performance, while analytic reasoning is reported as non-significant.
- Prompt-only debiasing is largely ineffective at mitigating motivated-reasoning effects in persona-assigned LLMs.

## How is this similar to GALILEO?

- If GALILEO evaluates/improves reasoning or factuality under different prompting conditions, this paper is direct evidence that “persona” prompt wrappers are not benign: they can systematically shift correctness in politically loaded reasoning and misinformation discernment.
- Reinforces the need for *controlled* prompt conditions and identity-neutral baselines when claiming robustness improvements.

## How is this different from GALILEO?

- This is a behavioral measurement/diagnosis study (bias under persona assignment), not a method for improving reasoning.
- Uses two specific human-study-inspired tasks (headline veracity + numeric evidence interpretation) rather than general-purpose reasoning benchmarks.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO includes explicit controls for prompt framing / role prompts / persona leakage, it can avoid the confound this paper highlights (persona-induced degradation).
- If GALILEO includes non-prompt interventions (training, retrieval, verification, tool use), it may outperform prompt-only “debiasing” shown ineffective here.

## Where GALILEO is weaker / needs to improve

- If GALILEO relies on persona/role prompting for behavior shaping (e.g., “as a <role>…”), it may inherit the same motivated-reasoning failure modes on identity-salient topics.
- If GALILEO reports aggregate accuracy without congruent-vs-incongruent slicing, it may miss the directional bias patterns emphasized here.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add an ablation: baseline vs persona-assigned vs “expert-role” prompts on any GALILEO factuality/reasoning evals.
- [ ] For politically/socially sensitive domains, report performance separately for identity-congruent vs identity-incongruent ground-truth conditions (where definable).
- [ ] In the paper, explicitly warn that persona prompting can *reduce* truthfulness/veracity and can create motivated-reasoning-like biases.
- [ ] If using prompt-based safety/debiasing, treat it as insufficient alone; consider stronger interventions (verification, retrieval, consistency checks, counterfactual prompting + adjudication).

## Quotes / details to potentially cite

- Abstract: persona-assigned LLMs show “up to 9% reduced veracity discernment” vs no-persona.
- Abstract: political personas are “up to 90% more likely” to correctly evaluate evidence on gun control when ground truth is congruent with induced identity.
- Abstract: “Prompt-based debiasing methods are largely ineffective at mitigating these effects.”
