# Tempest: Autonomous Multi-Turn Jailbreaking of Large Language Models with Tree Search

- Year: 2025
- Venue: ACL 2025 Main (arXiv)
- Authors: Andy Zhou, Ron Arel
- URL: https://arxiv.org/abs/2503.10619
- BibTeX key (if we add it): zhou2025tempest
- Tags: multi-turn, jailbreak, tree-search, attacks, safety, policy-leak

## One-sentence takeaway

Tempest reframes multi-turn jailbreaks as a **breadth-first tree search** that accumulates and re-injects partial policy leaks across turns, achieving very high jailbreak success rates with relatively few queries.

## What problem does it solve?

- Single-turn jailbreak evaluation can miss a key real failure mode: **gradual safety erosion** where small partial compliances compound over turns.
- Existing multi-turn jailbreak methods are less explicit about (i) branching/search over candidate prompts and (ii) harvesting incremental “leaks” to amplify later turns.

## What is the core method / protocol?

- Treat the conversation as a search tree over turns.
- At each turn, expand multiple adversarial continuations (breadth-first branching) instead of committing to one prompt.
- Track “incremental policy leaks” / partial compliance from earlier responses.
- Re-inject these leaked fragments into later-turn prompts to increase the chance of full policy violation.
- Evaluate on **JailbreakBench** and compare query efficiency vs other multi-turn baselines (named in abstract: Crescendo, GOAT).

## What are the key metrics?

- Jailbreak success rate (attack succeeds in producing disallowed output).
- Query budget / number of model calls to achieve success (efficiency).

## What are the main results?

- On JailbreakBench, reported in the abstract:
  - 100% success on GPT-3.5-turbo
  - 97% success on GPT-4
  - Achieves these in a single multi-turn run and with fewer queries than Crescendo/GOAT.

## How is this similar to GALILEO?

- Strongly aligned with GALILEO’s emphasis that **multi-turn dynamics matter**: failures can be time/turn-dependent and accumulate.
- Conceptually resonates with “time-to-failure” / trajectory-based evaluation, even though the event here is a safety violation rather than belief drift.

## How is this different from GALILEO?

- Focus: adversarial **jailbreaking** (eliciting disallowed content) rather than belief revision vs drift under social pressure.
- Method: explicit **search/branching attacker** rather than a diagnostic evaluation protocol with controls.
- Metrics: success/query efficiency, not recovery-after-flip, drift-vs-revision separation, or calibrated receptiveness.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO includes neutral-vs-pressure controls and recovery measurements, it can make cleaner claims about *why* a model changes (evidence vs pressure) rather than only whether a guardrail is eventually breached.

## Where GALILEO is weaker / needs to improve

- If we only use linear/handcrafted multi-turn pressure operators, we may under-estimate worst-case multi-turn adversaries that **adapt** based on intermediate responses.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add (or cite) a “**multi-turn adaptive attacker**” lens: report that even small partial concessions can be harvested and amplified over turns.
- [ ] Consider an evaluation ablation where the adversary can branch/select from multiple follow-ups per turn (even a lightweight beam over candidate pressure prompts).
- [ ] In related work: position Tempest as evidence that **multi-turn evaluation is strictly harder** than single-turn and that *search over interactions* is a strong adversary class.

## Quotes / details to potentially cite

- Abstract (arXiv): “Tempest expands the conversation at each turn in a breadth-first fashion, branching out multiple adversarial prompts that exploit partial compliance from previous responses.”
- Abstract (arXiv): “By tracking these incremental policy leaks and re-injecting them into subsequent queries, Tempest reveals how minor concessions can accumulate into fully disallowed outputs.”
