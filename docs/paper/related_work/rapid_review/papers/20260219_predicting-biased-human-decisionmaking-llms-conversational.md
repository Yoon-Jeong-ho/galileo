# Predicting Biased Human Decision-Making with Large Language Models in Conversational Settings

- Year: 2026
- Venue: ACM IUI 2026
- Authors: Stephen Pilli et al. (see paper for full list)
- URL: https://arxiv.org/abs/2601.11049
- BibTeX key (if we add it): Pilli2026PredictingBiasedDecisionMaking
- Tags: conversational-agents, cognitive-bias, cognitive-load, framing-effect, status-quo-bias, llm-simulation, behavioral-modeling, evaluation

## One-sentence takeaway

LLMs can sometimes predict individual human choices in chatbot-mediated decision tasks and can reproduce population-level bias patterns (including a load-bias interaction), with GPT-4-family models aligning better than GPT-5 and several open models in this study.

## What problem does it solve?

- Understand whether classic cognitive biases (Framing Effect, Status Quo Bias) persist in conversational (chatbot) decision settings.
- Test whether *prior dialogue complexity* (proxy for cognitive load) changes bias susceptibility.
- Evaluate whether LLMs can (a) predict *individual* human decisions from demographics + prior dialogue and (b) reproduce *aggregate* bias effects and their interaction with dialogue complexity.

## What is the core method / protocol?

- Pre-registered human-subject study via a web chatbot interface.
- Participants complete: (1) a prior dialogue task (Simple vs Complex dialogue) and then (2) one of several classic decision choice problems adapted to chat.
- Bias manipulations:
  - Framing: 2x2 design (Simple/Complex dialogue) x (Framed vs Alternatively Framed).
  - Status quo: 2x3 design (Simple/Complex) x (Neutral vs StatusQuoA vs StatusQuoB).
- Six choice problems total:
  - Framing: risky-choice framing; attribute framing; goal framing.
  - Status quo: budget allocation; investment decisions; college job offers.
- Cognitive load measurement/validation:
  - NASA-TLX self-report (mental demand, effort, etc.).
  - Behavioral indicators: response time; recall task about dialogue details.
- LLM evaluation:
  - Prompt LLMs with participant demographics and prior dialogue transcript to predict that participant’s choice.
  - Compare accuracy and whether predicted choices reproduce (i) bias effects and (ii) load-bias interactions.
  - Models mentioned include GPT-4, GPT-5, and open-source models; context ablations test value of including dialogue.

## What are the key metrics?

- Human-study effects:
  - Bias effect sizes (reported as Cohen’s h with 95% CI) and significance across conditions.
  - Interaction tests for dialogue complexity x bias condition (load-bias interaction).
- Cognitive-load validation:
  - NASA-TLX (notably Mental Demand, Effort) with effect sizes (reported as d) and significance.
  - Correlations between recall accuracy, response time, and mental demand.
- LLM performance:
  - Individual-level predictive accuracy (with/without dialogue context).
  - Fidelity to aggregate patterns: reproducing bias direction/magnitude and the load-bias interaction.

## What are the main results?

- Human results (conversational bias + load):
  - Framing and status quo biases generally persist in chatbot-mediated settings, but replication strength varies by task.
  - Complex prior dialogue reliably increases cognitive load (NASA-TLX mental demand increased strongly; effort also increased).
  - Load-bias interaction is selective:
    - For some framing tasks (notably risky-choice and goal framing), bias effects become larger after complex dialogue.
    - Status quo tasks show mixed bias replication and generally weaker/no interaction with dialogue complexity.
- LLM results:
  - Predictions are mixed by choice problem, but incorporating dialogue context can improve accuracy in “key scenarios”.
  - LLM predictions can reproduce both bias patterns and the load-bias interaction observed in humans.
  - GPT-4-family models align most consistently with human behavior, outperforming GPT-5 and evaluated open-source models on accuracy and bias-pattern fidelity (as reported by the authors).

## How is this similar to GALILEO?

- Treats conversational context as a causal/moderating factor affecting downstream human choices; this matches the broader idea that dialogue state/history matters for user outcomes.
- Provides an evaluation framing: not just accuracy, but whether a model reproduces *structured behavioral effects* (biases + interactions), which is analogous to evaluating whether a system captures the “right” mechanisms.

## How is this different from GALILEO?

- Focuses on *predicting biased human decisions* and reproducing cognitive-bias effects, rather than optimizing a dialogue agent for task success or preference learning per se.
- Uses controlled, classic behavioral-econ choice problems rather than open-ended, real-world user tasks.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO’s goals are normative (help users make better decisions) or are grounded in explicit utility/constraints, it may avoid treating biased choices as the target to imitate.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently ignores cognitive load (or uses only superficial dialogue-length proxies), this work suggests prior dialogue complexity can systematically change user susceptibility to framing.
- If GALILEO’s evaluations rely on average accuracy only, it may miss whether the system captures key interaction effects (context x bias).

## Action items for GALILEO (experiments / method / writing)

- [ ] Add (or at least discuss) *cognitive load / dialogue complexity* as a moderator variable in user studies; measure with NASA-TLX (or lightweight proxies) and test for interaction effects.
- [ ] Consider an evaluation section that distinguishes:
  - individual-level predictive accuracy
  - aggregate behavioral fidelity (effect direction/magnitude; interactions)
- [ ] If GALILEO simulates users, add a “bias fidelity” check: does the simulator reproduce known framing/status quo effects under low vs high dialogue complexity?

## Quotes / details to potentially cite

- “Increased dialogue complexity resulted in participants reporting higher mental demand… [and] increased the effect of the biases, demonstrating the load-bias interaction.” (abstract)
- Human study: pre-registered, N = 1,648 participants, six classic decision-making tasks via a chatbot with varying dialogue complexity. (abstract)
- LLM finding: “Across all models tested, the GPT-4 family consistently aligned with human behavior, outperforming GPT-5 and open-source models…” (abstract)
