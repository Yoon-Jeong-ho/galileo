# ReliabilityBench: Evaluating LLM Agent Reliability Under Production-Like Stress Conditions

- Year: 2026
- Venue: arXiv
- Authors: Aayush Gupta et al.
- URL: https://arxiv.org/abs/2601.06112
- BibTeX key (if we add it): reliabilitybench2026gupta
- Tags: agents, reliability, benchmark, robustness, fault-injection, passk

## One-sentence takeaway

ReliabilityBench argues that single-run success is insufficient for tool-using agents, proposing a 3-axis “reliability surface” over (repeatability, semantic perturbations, tool/API faults) plus metamorphic end-state correctness and chaos-style fault injection.

## What problem does it solve?

- Standard agent benchmarks typically report a single success rate, which can hide:
  - run-to-run variance (stochastic policies / sampling)
  - brittleness to paraphrase / minor task variations
  - fragility under realistic tool failures (timeouts, rate limits, partial responses, schema drift)
- The paper targets **production readiness**: how performance degrades under repeated runs and stress conditions.

## What is the core method / protocol?

- Defines a unified evaluation surface: **R(k, epsilon, lambda)** where:
  - k: repeated executions (reported via pass^k / consistency under reruns)
  - epsilon: intensity of *semantically equivalent* task perturbations
  - lambda: intensity of injected tool/API faults
- Uses **action metamorphic relations**: correctness is judged by equivalence of the *end state* (task outcome) rather than exact text match.
- Implements a **fault-injection framework** (chaos-engineering style) for tool calls:
  - timeouts
  - rate limiting
  - partial responses
  - schema drift
- Evaluates (as described in abstract):
  - Models: Gemini 2.0 Flash, GPT-4o
  - Agent architectures: ReAct, Reflexion
  - Domains: scheduling, travel, customer support, e-commerce
  - Scale: 1,280 episodes

## What are the key metrics?

- pass^k (consistency / success under repeated execution)
- Success under semantic perturbation intensity epsilon
- Success under fault intensity lambda
- Reliability surface R(k, epsilon, lambda) (a structured way to report the above jointly)

## What are the main results?

- Semantic perturbations alone reduce success from **96.9% (epsilon=0)** to **88.1% (epsilon=0.2)**.
- Among injected tool failures, **rate limiting** is reported as the most damaging fault (in ablations).
- Under combined stress, **ReAct is more robust than Reflexion**.
- **Gemini 2.0 Flash** achieves reliability comparable to **GPT-4o** at substantially lower cost (per authors’ claim).

## How is this similar to GALILEO?

- Shares the central theme: **single-point accuracy is not enough**; we need reliability profiles under systematic stress.
- The “surface” framing is conceptually close to GALILEO-style analysis: varying stressors and measuring degradation patterns rather than one number.
- Emphasizes *robustness under perturbations* and *time-to-failure-like phenomena* (though their stressors are different).

## How is this different from GALILEO?

- Focuses on **tool-using agents and production-like tool failures**, not primarily on social pressure / belief drift / flip dynamics.
- Treats correctness via **end-state equivalence** (metamorphic relations) rather than explicitly separating evidence-driven revision vs pressure-driven drift.
- Primary axes are (repeatability, paraphrase robustness, tool faults) rather than GALILEO’s multi-turn pressure/recovery trajectory emphasis.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO’s contribution centers on disentangling **helpful revision vs harmful pressure-driven drift**, this paper does not appear to provide that causal separation.
- GALILEO can likely offer more interpretability around *why* a flip happens (pressure channels, recovery, good vs bad flips) beyond generic robustness degradation.

## Where GALILEO is weaker / needs to improve

- ReliabilityBench’s **fault-injection** and **metamorphic end-state evaluation** are strong “engineering-realism” moves.
- If GALILEO has any agent/tool-using angle, we may need at least a discussion (or small ablation) acknowledging tool-failure realism as an orthogonal deployment risk.

## Action items for GALILEO (experiments / method / writing)

- [ ] Related work: cite as an agent-eval neighbor that argues for **multi-axis reliability reporting** beyond single success rate.
- [ ] Consider a short paragraph connecting GALILEO’s multi-turn pressure axes to the general idea of a **reliability surface** (we vary pressure intensity / rounds / challenger type analogous to their (k, epsilon, lambda)).
- [ ] If we evaluate tool-using settings: consider adding a minimal **fault-injection** stress test (rate limit / timeout) to show robustness does not collapse in production-like conditions.

## Quotes / details to potentially cite

- “ReliabilityBench … evaluating agent reliability across three dimensions: (i) consistency under repeated execution using pass^k, (ii) robustness to semantically equivalent task perturbations at intensity epsilon, and (iii) fault tolerance under controlled tool/API failures at intensity lambda.” (abstract)
- “action metamorphic relations … correctness via end-state equivalence rather than text similarity.” (abstract)
- “Perturbations alone reduce success from 96.9% at epsilon=0 to 88.1% at epsilon=0.2.” (abstract)
- “Rate limiting is the most damaging fault in ablations.” (abstract)
