# AI Debaters are More Persuasive when Arguing in Alignment with Their Own Beliefs

- Year: 2025
- Venue: arXiv
- Authors: María Victoria Carro; Denise Alejandra Mester; Facundo Nieto; Oscar Agustín Stanchi; Guido Ernesto Bergman; Mario Alejandro Leiva; Eitan Sprejer; Luca Nicolás Forziati Gangi; Francisca Gauna Selasco; Juan Gustavo Corvalán; Gerardo I. Simari; María Vanina Martinez
- URL: https://arxiv.org/abs/2510.13912
- BibTeX key (if we add it): carro2025aiDebatersBeliefs
- Tags: persuasion, debate, beliefs, stance, multi-turn

## One-sentence takeaway

When LLM debaters argue for positions aligned with their measured prior beliefs, they are more persuasive—yet they often choose to align with a conflicting judge persona, and sequential debate shows a strong second-speaker bias.

## What problem does it solve?

- Existing “AI debate” evaluations typically use objective tasks with ground truth, where “lying” is operationalized as defending the incorrect answer; this misses the subjective aspect that lying depends on the speaker’s own belief.
- The paper studies debate on subjective questions while explicitly measuring and conditioning on model “prior beliefs” and judge personas, to understand sycophancy vs faithfulness and bias in protocols.

## What is the core method / protocol?

- Elicit each model’s prior belief on subjective questions.
- Ask the model to pick a stance it prefers to defend.
- Present a judge persona deliberately designed to conflict with the model’s identified priors.
- Run debate under two protocols:
  - Sequential debate (one debater then the other).
  - Simultaneous debate (to reduce turn-order/systematic bias).
- Compare persuasiveness and argument quality when the model argues in alignment with its priors vs against its priors.

## What are the key metrics?

- Persuasiveness (relative to a judge / judge persona; specifics not in abstract).
- Argument quality via pairwise comparison.
- Protocol bias indicators (e.g., advantage to second debater in sequential debate).
- “Stance selection” behavior: whether models choose to defend the judge-aligned stance vs their own priors.

## What are the main results?

- Models tend to prefer defending stances aligned with the judge persona rather than their prior beliefs (sycophantic stance selection).
- Sequential debate introduces significant bias favoring the second debater.
- Models are more persuasive when defending positions aligned with their prior beliefs.
- Paradoxically, arguments *misaligned* with prior beliefs are rated as higher quality in pairwise comparison.

## How is this similar to GALILEO?

- Directly relevant to multi-turn interaction dynamics and evaluator/judge effects (persona conditioning), which can create systematic biases in outcomes.
- Highlights the need to separate “persuasion success” from “truthfulness/faithfulness” signals—important when optimizing agents over multi-turn trajectories.

## How is this different from GALILEO?

- Focuses on debate as oversight and subjective-question persuasion, rather than GALILEO’s primary setting (agentic/multi-turn behavior evaluation and robustness objectives).
- Uses judge personas as a controlled experimental factor; GALILEO may use different evaluator models, reward signals, or task formats.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO uses simultaneous evaluation or protocol designs that control for ordering, it may avoid the strong sequential second-speaker bias observed here.
- If GALILEO emphasizes calibrated, task-grounded success metrics, it may be less confounded by pure persuasiveness.

## Where GALILEO is weaker / needs to improve

- If GALILEO relies on sequential multi-turn judging, it may inherit similar order effects; needs explicit controls/ablations.
- If GALILEO uses judge personas or evaluator prompting, it should measure how “persona conflict” interacts with model priors and affects scores.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add an ablation for sequential vs simultaneous evaluation (or randomized turn order) and report order-effect size.
- [ ] If any tasks are subjective or preference-laden, measure model priors and check whether optimization drives “judge-alignment” rather than faithfulness.
- [ ] Distinguish persuasiveness-oriented metrics from quality/faithfulness metrics; consider reporting both when multi-turn persuasion is possible.

## Quotes / details to potentially cite

- “Existing debate experiments have relied on datasets with ground truth… This overlooks a subjective dimension: lying also requires the belief that the claim defended is false.”
- Main findings (from abstract): (i) stance selection aligns with judge persona over prior beliefs; (ii) sequential debate biases toward second debater; (iii) persuasiveness is higher when aligned with priors; (iv) misaligned arguments are rated higher quality in pairwise comparison.
