# PTCBench: Benchmarking Contextual Stability of Personality Traits in LLM Systems

- Year: 2026
- Venue: arXiv
- Authors: Yuhan Ma; Xiaoyu Zhang; Junjie Wang; Qiang Hu; Chao Shen; Xiaofei Xie
- URL: https://arxiv.org/abs/2602.00016
- BibTeX key (if we add it): ptcbench2026
- Tags: personality, big-five, benchmark, context, stability, agents

## One-sentence takeaway

PTCBench measures how much an LLM/agent’s Big-Five personality profile shifts when the same “person” is placed into controlled location/life-event contexts, revealing sizable and scenario-specific trait drift (especially for agentic systems).

## What problem does it solve?

- Current “LLM personality” evaluations are mostly static (single setting) or focus on persona/role consistency, but human personality is known to vary systematically with situation and life events.
- For deployed affective agents/companions, uncontrolled personality drift can harm trust, coherence, and perceived authenticity.
- Need a repeatable benchmark to quantify context-induced personality trait change and compare models/agent frameworks.

## What is the core method / protocol?

- Define a benchmark (PTCBench) with 12 external conditions spanning:
  - location contexts, and
  - life events (the paper highlights negative events like Divorce / Unemployment as high-impact).
- Measure personality using the NEO Five-Factor Inventory (NEO-FFI) along O/C/E/A/N.
- Evaluate models “before vs after” introducing each condition; aggregate trait records and analyze variability patterns.
- Study includes both foundation LLMs and agentic systems (paper mentions AutoGen as an example agent).

## What are the key metrics?

- Trait scores for Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism (NEO-FFI).
- Trait change / deviation magnitude across contexts (implied: pre/post deltas; variability across scenarios).
- Secondary observation: correlation with reasoning behavior changes under some contexts.

## What are the main results?

- Collected 39,240 personality trait records across four LLMs and two agents.
- Some foundation models are comparatively stable (example: Gemini-2.0-Flash), while agentic systems (example: AutoGen) show amplified trait variability under negative life events and task-oriented contexts.
- Baseline personality settings modulate the extent of trait change.
- Divorce and Unemployment produce the largest deviations and are associated with measurable changes in reasoning behavior.

## How is this similar to GALILEO?

- Both care about robustness/consistency of agent behavior under changing context.
- Both motivate evaluation under “realistic, evolving environments” rather than single-shot tests.

## How is this different from GALILEO?

- PTCBench is explicitly about *personality trait dynamics* (Big Five/NEO-FFI) under contextual interventions, not task performance per se.
- Uses psychometric questionnaires (NEO-FFI-style) rather than domain/task-specific metrics.
- Focuses on trait drift and psychological alignment; GALILEO may emphasize capability/goal performance and generalization under context.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO’s evaluation is grounded in task outcomes and reliability (not only self-reported questionnaire answers), it may be less vulnerable to prompt-format artifacts.
- If GALILEO has stricter control of tools/memory/interaction loop, it may better isolate causal factors behind behavior changes.

## Where GALILEO is weaker / needs to improve

- If GALILEO does not explicitly measure “persona/personality stability” dimensions, PTCBench highlights a missing axis that matters for companion-style deployments.
- May need a principled notion of “acceptable behavioral drift” vs “inconsistency” as the environment changes.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a small “personality/affect stability under context” slice to the evaluation suite (even if lightweight), inspired by PTCBench’s pre/post context design.
- [ ] Consider reporting robustness separately for (a) base model vs (b) agentic orchestration (memory/tools), since agent layers may amplify drift.
- [ ] In related work, position GALILEO against “context-induced trait drift” benchmarks to argue why context robustness should cover more than persona consistency.

## Quotes / details to potentially cite

- “PTCBench subjects models to 12 distinct external conditions spanning diverse location contexts and life events, and rigorously assesses the personality using the NEO Five-Factor Inventory.”
- “Our study on 39,240 personality trait records reveals that certain external scenarios (e.g., ‘Unemployment’) can trigger significant personality changes of LLMs, and even alter their reasoning capabilities.”
- Reported findings summary: agentic systems (e.g., AutoGen) show amplified variability; baseline personality settings modulate trait change; Divorce/Unemployment yield largest deviations.
