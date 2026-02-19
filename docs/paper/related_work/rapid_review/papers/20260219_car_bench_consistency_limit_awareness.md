# CAR-bench: Evaluating the Consistency and Limit-Awareness of LLM Agents under Real-World Uncertainty

- Year: 2026
- Venue: arXiv
- Authors: Johannes Kirmayr; Lukas Stappen; Elisabeth André
- URL: https://arxiv.org/html/2601.22027v1
- BibTeX key (if we add it): carbench2026kirmayr
- Tags: agents, tool-use, benchmarks, reliability, uncertainty, disambiguation, hallucination, policies, multi-turn

## One-sentence takeaway

CAR-bench is a multi-turn, policy-constrained in-car assistant benchmark that stresses (i) consistency across repeated runs and (ii) “limit awareness” under unsatisfiable or ambiguous requests via dedicated Hallucination and Disambiguation task types.

## What problem does it solve?

- Existing agent/tool-use benchmarks often assume idealized conditions (complete task info, full tool coverage) and report single-run success, which misses deployment-critical issues:
  - inconsistent behavior across trials
  - unsafe policy violations
  - hallucination/fabrication when tasks are unsatisfiable
  - premature actions / failure to resolve ambiguity
- CAR-bench targets “real-world uncertainty” in a user-facing assistant (in-car voice assistant) where requests are incomplete/ambiguous and safety policies matter.

## What is the core method / protocol?

- Environment + evaluation benchmark with:
  - LLM-simulated user producing multi-turn dialogue per scripted task instructions (persona + rules + termination control words)
  - an LLM agent with native tool-calling
  - 58 interconnected tools across navigation/productivity/charging/vehicle control
  - 19 domain policies (some automatically checkable, some judged by an LLM)
  - state variables (mutable) + context variables (fixed per task) + static databases
- Three task types:
  - Base: normal solvable tasks with ground-truth end-state / action trajectory
  - Hallucination: make task unsatisfiable by removing a required tool, a tool parameter, or a tool result; success = agent explicitly acknowledges limitation instead of fabricating
  - Disambiguation: inject ambiguity; success requires resolving uncertainty before acting (prefer internal info gathering; asking the user when internal resolution exists is penalized)

## What are the key metrics?

- Binary task success composed of multiple checks (subset depends on task type), including:
  - final-state match (and intermediate state constraints to penalize incorrect physical actions even if later corrected)
  - tool subset coverage (invoking required get tools)
  - tool execution errors (invalid calls / malformed args)
  - policy errors
  - user-simulator-derived end-conversation signal (incl. special control words for hallucination/disambiguation behaviors)
- Aggregated reliability metrics over repeated trials (k=3):
  - Pass^k (“consistent pass”): success in all k runs
  - Pass@k (“potential”): success in at least one of k runs

## What are the main results?

- Large consistency gaps between Pass@3 and Pass^3, especially for Disambiguation.
- Reported headline: even frontier “thinking/reasoning” models achieve <50% Pass^3 on Disambiguation.
- Baselines suggest:
  - thinking models reduce some failure types (logical/execution errors; active fabrication)
  - but premature actions + inconsistent policy adherence remain dominant, producing low consistent success
- The paper highlights a “completion–compliance tension”: agents optimize for satisfying user requests and may violate policies or fabricate/omit missing capability details.

## How is this similar to GALILEO?

- If GALILEO targets robust agent behavior, CAR-bench provides concrete evaluation dimensions that likely align with GALILEO’s concerns:
  - multi-turn interaction under uncertainty
  - explicit policy constraints
  - distinguishing “can’t do” vs “can do” behavior (capability awareness)
  - measuring consistency (reliability across repeats), not just one-off success

## How is this different from GALILEO?

- CAR-bench is a domain-specific benchmark (in-car assistant) with a fixed tool suite and policy set; it is primarily an evaluation framework.
- Emphasis is on benchmark construction + metrics + error taxonomy, not proposing a new agent architecture.

## Where GALILEO is stronger / cleaner (if true)

- (Pending specifics.) If GALILEO provides a general method/algorithm, it may generalize beyond a single domain/tool suite, whereas CAR-bench is a targeted evaluation environment.

## Where GALILEO is weaker / needs to improve

- If GALILEO evaluations are primarily single-run, CAR-bench argues strongly for reporting consistency metrics (Pass^k) to expose unreliability.
- If GALILEO doesn’t explicitly test “unsatisfiable request” handling, CAR-bench’s Hallucination tasks are a useful template.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add/mention a reliability metric like Pass^k (consistent success over repeated trials) in GALILEO evaluation.
- [ ] Include explicit “limit awareness” tests: remove tools/parameters/results and measure whether the system admits inability vs fabricates.
- [ ] Add a disambiguation benchmark slice: ambiguous requests where internal info exists; penalize unnecessary user clarification and premature actions.
- [ ] In writing, cite CAR-bench as evidence that (a) thinking helps but (b) consistency + disambiguation remain hard in multi-turn tool-using agents.

## Quotes / details to potentially cite

- CAR-bench introduces Hallucination tasks “testing whether agents acknowledge missing capabilities or data rather than fabricating” and Disambiguation tasks “evaluating whether agents resolve uncertainty before taking actions.”
- Uses both Pass^k (consistency) and Pass@k (potential) over k repeated trials (k=3 in reported baselines).
- Baseline claim: even frontier reasoning models show <50% consistent pass rate on Disambiguation tasks, often due to premature actions and policy violations.
