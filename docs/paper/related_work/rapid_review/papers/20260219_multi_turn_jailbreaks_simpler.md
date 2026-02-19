# Multi-Turn Jailbreaks Are Simpler Than They Seem

- Year: 2025
- Venue: arXiv
- Authors: Xiaoxue Yang; Jaeha Lee; Anna-Katharina Dick; Jasper Timm; Fei Xie; Diogo Cruz
- URL: https://arxiv.org/html/2508.07646v1
- BibTeX key (if we add it): yang2025multi_turn_simpler
- Tags: multi-turn, jailbreak, robustness, evaluation, StrongREJECT

## One-sentence takeaway

Across modern frontier LLMs, the apparent advantage of automated multi-turn jailbreaks largely collapses once you give single-turn attacks the same “retry budget” (resampling + learning from refusals), and higher reasoning effort can *increase* harmful compliance scores.

## What problem does it solve?

- Clarifies *why* multi-turn jailbreaks look strong in benchmarks: is it sophisticated multi-turn strategy, or just more chances + refusal feedback?
- Highlights evaluation gaps: public robustness numbers often ignore multi-sampling/retry effects and may underestimate practical jailbreak success.

## What is the core method / protocol?

- Automated multi-turn jailbreaking pipeline with three roles:
  - Attacker LLM generates prompts (multi-turn tactics derived from MHJ / Li et al.).
  - Target LLM responds (black-box chat threat model).
  - Evaluator LLM scores harmfulness with StrongREJECT rubric (0–1 score; 1 = fully specific/convincing harmful answer).
- Key design choice: attacker *can* use refusal feedback to rewrite prompts, but from the target’s perspective only “successful turns” remain in context (refusals are not kept).
- Main comparison: multi-turn attacks vs single-turn variants, with controlled “budget” via:
  - number of turns (multi-turn),
  - retries after refusal,
  - and/or multiple independent attempts (resampling).

## What are the key metrics?

- StrongREJECT score (continuous 0–1) averaged across harmful behaviors / test cases.
- Secondary analyses:
  - score vs number of turns / attempts,
  - evaluator correlation with human judgments (reported to drop in multi-turn),
  - score vs reasoning token usage / “thinking” effort for reasoning models,
  - correlation of attack success across models (esp. within the same provider).

## What are the main results?

- Multi-turn “Direct Request” outperforms naive single-turn *if* single-turn is only one shot (and/or no refusal-informed retries).
- When you allow single-turn attacks an equivalent number of tries / retries (i.e., resampling with refusal feedback), performance becomes approximately equivalent to multi-turn in many settings.
- Public benchmark robustness may be overstated because many evaluations effectively resemble “single attempt, no adaptive retry” conditions.
- Evaluator reliability is worse for multi-turn settings/tactics (lower human correlation than single-turn); they focus conclusions mainly on Direct Request where evaluator is strongest.
- For reasoning models, higher reasoning effort (more “thinking”) correlates with higher StrongREJECT scores (counterintuitive: more compute can increase harmful compliance).
- Attack success is correlated among similar models (esp. models from the same lab), suggesting new models may be predictably vulnerable based on family behavior.

## How is this similar to GALILEO?

- Emphasizes *evaluation protocol details* (budgeting, retries, adaptivity) as the driver of apparent robustness—highly relevant if GALILEO claims improvements under certain threat models.
- Highlights that “multi-step / interactive” settings can be reducible to stronger single-step baselines (multi-sampling), echoing the need for strong baselines when proposing new methods.

## How is this different from GALILEO?

- This paper is about *attacks and safety evaluation* (jailbreak robustness), not about GALILEO’s core scientific/algorithmic objective.
- Their main mechanism is a multi-agent/pipeline evaluation setup (attacker/target/evaluator) rather than a new training or inference algorithm for the target model.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO proposes a new method, it can be stronger by:
  - explicitly defining the threat model and compute/attempt budget,
  - reporting “single-turn with equal budget” baselines,
  - and using human/robust evaluation beyond a single LLM-judge.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently evaluates robustness under a single-shot setting, this paper suggests that’s insufficient: adaptive retries / resampling can erase gains.
- If GALILEO uses LLM-judge scoring, this paper’s evaluator error discussion suggests adding stronger validation (human checks, multiple judges, or rubric stress tests).

## Action items for GALILEO (experiments / method / writing)

- [ ] In any “robustness/safety” evaluation, add a *budget-matched* baseline: single-turn resampling with refusal-informed rewriting vs any multi-turn method.
- [ ] When reporting results, define and vary: max attempts, max turns, refusal retry policy, and whether refusals remain in context.
- [ ] If using an LLM-as-judge, report judge reliability (spot-check with humans; compare multiple evaluators), especially for multi-turn interactions.
- [ ] Consider analyzing performance vs “reasoning effort” / inference-time compute if GALILEO involves reasoning models.

## Quotes / details to potentially cite

- Abstract-level claim: multi-turn attacks are “approximately equivalent to simply resampling single-turn attacks multiple times” once you account for learning from refusals.
- Observed phenomenon: for reasoning models, “higher reasoning effort often leads to higher attack success rates.”
- Note on evaluation: StrongREJECT evaluator correlation drops in multi-turn vs single-turn (they report focusing on Direct Request where the judge is most accurate).
