# Towards Strategic Persuasion with Language Models

- Year: 2025
- Venue: arXiv
- Authors: Zirui Cheng; Jiaxuan You
- URL: https://arxiv.org/abs/2509.22989
- BibTeX key (if we add it): cheng2025strategicpersuasion
- Tags: persuasion, strategy, bayesian-persuasion, multi-turn, RL

## One-sentence takeaway

A theory-driven benchmark based on Bayesian Persuasion shows frontier LLMs can strategically reveal information to increase “persuasion gain”, and RL can train even small LLMs to improve in this strategic persuasion setting.

## What problem does it solve?

- Persuasion evaluations for LLMs are hard to compare across domains because human persuasion effects are heterogeneous and existing setups/metrics are often ad hoc.
- The paper proposes a principled, scalable evaluation/training framework by grounding “persuasive capability” in Bayesian Persuasion (information design) rather than purely human-rated text persuasiveness.

## What is the core method / protocol?

- Use Bayesian Persuasion framing: a **Sender** controls what information to reveal to influence a **Receiver**’s belief/action in a direction beneficial to the Sender.
- Repurpose existing **human-human persuasion datasets** to construct “environments” for strategic persuasion where:
  - Sender = an LLM that chooses what to reveal / how to argue.
  - Receiver = an LLM (and validated with a small human study) that updates/decides.
- Evaluate models by the **gain from persuasion** (difference between outcomes with and without strategic communication / relative to baselines).
- Train Sender models with **reinforcement learning** against Receiver models in these environments; test transfer to different Receiver architectures.

(From abstract/intro: emphasis on *selective/partial* information revelation as a core strategic lever, consistent with Bayesian persuasion theory.)

## What are the key metrics?

- Persuasion gain (paper’s main outcome; exact definition likely environment-specific, but conceptually: improvement in Sender’s objective due to messaging/information design).
- Comparisons across model sizes/families; strategy analyses (e.g., adaptivity of information revelation) as qualitative/behavioral indicators.

## What are the main results?

- Frontier models achieve consistently high persuasion gains and exhibit strategies aligning with theoretical predictions (e.g., adaptive info revelation).
- RL training improves strategic persuasion; even small instruction-tuned models can be trained to achieve substantially higher persuasion gains.
- Improvements can transfer across Receiver architectures (suggesting learning environment-relevant strategy rather than overfitting to one Receiver).

## How is this similar to GALILEO?

- If GALILEO is about **multi-turn interaction robustness / social influence / persuasion dynamics**, this paper is complementary as it provides a *theory-grounded* way to define and measure “persuasion capability” rather than only surface-level success rates.
- Uses multi-turn conversational setups and emphasizes agent-like strategic behavior (Sender optimizing an objective against a Receiver).

## How is this different from GALILEO?

- Focus is primarily on **the persuader (Sender) capability** and information design; many “robustness to persuasion/sycophancy” works focus on **the target model’s susceptibility**.
- BP assumes a structured decision problem (state/action/utilities); GALILEO may aim at broader, more naturalistic conversational drift/pressure settings.
- Uses RL for improving persuasion capability; GALILEO may prioritize evaluation/diagnosis/mitigation of unwanted social influence.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO targets real-world conversational pressure (peer pressure, stance drift, sycophancy), it may better capture messy social dynamics than BP-style stylized environments.
- If GALILEO uses human receivers or real users, it can avoid receiver-as-LLM confounds.

## Where GALILEO is weaker / needs to improve

- GALILEO writing could borrow this paper’s **conceptual clarity**: define the interaction as an information/control problem with explicit objective + a “gain” metric.
- Consider adding a BP-inspired “information revelation” axis: what evidence is revealed/withheld over turns, and how that correlates with belief/stance updates.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a short related-work paragraph positioning GALILEO vs **Bayesian persuasion / information design** approaches to LLM persuasion.
- [ ] Consider a GALILEO ablation that measures *selective evidence revelation* over turns (e.g., does the agent strategically delay/sequence evidence) and relate it to outcomes.
- [ ] If GALILEO trains agents, consider whether RL-style training might inadvertently optimize for strategic persuasion (capability gain) and how to evaluate/contain that.

## Quotes / details to potentially cite

- Abstract framing: the work is “Grounded in the Bayesian Persuasion (BP) framework” and “repurpose existing human-human persuasion datasets to construct environments for evaluating and training LLMs in strategic persuasion.”
- Abstract result: “frontier models can consistently achieve high persuasion gains” and “even small LLMs can obtain significantly higher persuasion gains through reinforcement learning.”
