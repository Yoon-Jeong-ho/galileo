# Implicit Probabilistic Reasoning Does Not Reflect Explicit Answers in Large Language Models

- Year: 2024 (v1; latest arXiv v4 Feb 2026)
- Venue: Transactions on Machine Learning Research (per arXiv)
- Authors: Manuel Mondal; Ljiljana Dolamic; Gérôme Bovet; Philippe Cudré-Mauroux; Julien Audiffren
- URL: https://arxiv.org/abs/2406.14986
- BibTeX key (if we add it): Mondal2024ImplicitProbReasoning
- Tags: implicit-evaluation, next-token-probabilities, calibration, probabilistic-reasoning, stated-vs-revealed, robustness

## One-sentence takeaway
Even when LLMs answer probability MCQs correctly, their *next-token probability mass* during equivalent text-completion scenarios can strongly diverge from the true outcome likelihoods, revealing a gap between “explicit answers” and “implicit” generative behavior.

## What problem does it solve?
- Standard evaluations of probabilistic reasoning often use explicit QA / MCQs, but these can be brittle (e.g., answer-order bias) and may not reflect how models actually behave during free-form generation.
- The paper targets a measurement gap: how to test whether probabilistic information in context is genuinely integrated into the model’s generative distribution.

## What is the core method / protocol?
- Define two paradigms:
  - **Explicit probabilistic reasoning**: ask an MCQ and score the selected option.
  - **Implicit probabilistic reasoning**: rephrase the same scenario as a **text-completion** with a fixed set of possible outcomes; evaluate the model by inspecting its **next-token probability** assigned to each outcome token.
- Compare the model’s predicted distribution (or at least argmax outcome / ranking) against the known ground-truth likelihoods implied by the scenario.
- The paper highlights cases where priors / irrelevant context features improperly influence the implicit next-token distribution.

## What are the key metrics?
- Agreement between model next-token probability assignments and the true outcome distribution (often practically evaluated via ranking / maximum-likelihood outcome alignment rather than full calibration).
- Contrast between performance on explicit MCQs vs implicit completion-based scoring.

## What are the main results?
- Models can show “solid” performance on **explicit** MCQs, but their **implicit** probabilistic reasoning during text completion can diverge substantially from ground truth.
- Implicit predictions are shown to be sensitive to factors that *should not* change the correct probabilistic outcome (as described in the abstract), e.g.:
  - independent prior events,
  - partial observations,
  - statistical background information.
- Conclusion: MCQ-style correctness may overestimate how reliably models will integrate probabilistic information in generation.

## How is this similar to GALILEO?
- Shares the core concern that **single-turn explicit answers** can be misleading about the model’s underlying state/behavior.
- Methodologically resonates with GALILEO’s emphasis on **protocol-level measurement** (not just end accuracy), and with diagnosing “drift”/instability that emerges in more realistic interaction/generation settings.

## How is this different from GALILEO?
- Not about social pressure / persuasion / persona-driven multi-turn dynamics; focuses on probabilistic reasoning and the mismatch between explicit QA and implicit next-token distributions.
- The “multi-turn” aspect here is not the central experimental manipulation (unlike GALILEO’s sustained pressure/control over turns).

## Where GALILEO is stronger / cleaner (if true)
- GALILEO directly targets **interaction dynamics** (survival / TOF / recovery) under controlled multi-turn protocols with a neutral re-asking control to separate drift vs pressure.
- GALILEO’s metrics are explicitly temporal (turn-indexed) and directly tied to ground-truth task correctness under repeated challenge.

## Where GALILEO is weaker / needs to improve
- GALILEO largely treats the model as a black box producing discrete answers; this paper suggests an additional diagnostic axis: **token-probability-level evidence** about whether the model’s generation distribution is consistent with its explicit answer.

## Action items for GALILEO (experiments / method / writing)
- [ ] Related work framing: cite as evidence that “explicit answer accuracy” may not reflect underlying generative competence; motivates measuring stability under interaction.
- [ ] Optional appendix idea (if feasible): for a subset of tasks, log **probability mass on the correct answer token/option** across turns to complement survival/TOF (only if already accessible in the inference stack).

## Quotes / details to potentially cite
- From the abstract: they “rephrase MCQs as text-completion scenarios … and compare the model's next-token probability assignments to the true likelihood of the outcomes,” finding that implicit probabilistic reasoning “often significantly diverge[s] from the known ground truth,” and is “improperly influenced” by unrelated factors (prior events / partial observations / background statistics).
