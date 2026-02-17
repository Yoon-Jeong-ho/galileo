# When LLMs get significantly worse: A statistical approach to detect model degradations

- Year: 2026
- Venue: ICLR 2026 (arXiv)
- Authors: Jonas Kübler, Kailash Budhathoki, Matthäus Kleindessner, Xiong Zhou, Junming Yin, Ashish Khetan, George Karypis
- URL: https://arxiv.org/abs/2602.10144
- BibTeX key (if we add it): kubler2026llms_get_significantly_worse
- Tags: monitoring, degradation-detection, drift, evaluation, statistics, nondeterminism

## One-sentence takeaway

A practical hypothesis-testing framework (McNemar-style paired test) for deciding whether small observed accuracy drops after inference/optimization changes are real degradations vs evaluation noise, with sensitivity down to ~0.3% deltas.

## What problem does it solve?

- When deploying “optimized” inference stacks (kernels, precision choices, quantization, speculative decoding, etc.), even *theoretically lossless* changes can lead to nondeterministic output differences (e.g., floating-point non-associativity) and tiny accuracy fluctuations.
- Standard reporting of aggregate benchmark accuracy doesn’t directly tell you if a small drop is *statistically meaningful* (true degradation) or just finite-sample / run-to-run noise.

## What is the core method / protocol?

- Treat baseline model/system A vs candidate model/system B as *paired* on the **same evaluation items**.
- For each item i, compute correctness indicators (0/1): A_i, B_i.
- Use a McNemar-test-style hypothesis test on the discordant pairs (A_i=1,B_i=0 vs A_i=0,B_i=1) to test whether B is significantly worse than A while controlling false positive rate.
- Key insight emphasized: compare **per-sample outcomes**, not only task-level aggregated scores.
- Multi-benchmark aggregation: proposes three ways to combine evidence across datasets/benchmarks into a single “degraded vs not degraded” decision (details not fully reviewed here).
- Provides tooling integrated with LM Evaluation Harness + a reference implementation.

## What are the key metrics?

- Primary: accuracy (binary correctness) per example; test decision based on discordant counts.
- Statistical: p-values / decision thresholds with controlled false positive rate (one-sided “is worse” framing).
- Practical sensitivity: smallest detectable degradation at a given sample size / benchmark suite.

## What are the main results?

- Demonstrates that the test flags genuinely degraded variants while not flagging optimizations argued to be provably lossless.
- Empirically, even ~0.3% accuracy drops can be identified as statistically significant (given enough paired samples).
- Includes discussion/examples of nondeterminism at temperature 0 due to numerical effects in real inference stacks.

## How is this similar to GALILEO?

- Shared theme: *robust evaluation under instability*—distinguishing real model changes from artifacts/noise.
- Relevant to GALILEO’s broader “drift/instability” story: evaluation pipelines themselves introduce nondeterminism, so statistical rigor matters when claiming small effect sizes.

## How is this different from GALILEO?

- Focuses on **single-turn benchmark accuracy degradation detection** (paired hypothesis testing), not multi-turn interaction dynamics (e.g., turn-of-failure/survival analysis) or persuasion/sycophancy.
- Concerned with changes induced by inference/optimization/hardware/software stacks rather than conversational pressure/adversarial dialogue.

## Where GALILEO is stronger / cleaner (if true)

- GALILEO targets *behavioral robustness across turns* and can surface failure dynamics (when/why models drift), not just whether aggregate accuracy changed.

## Where GALILEO is weaker / needs to improve

- If GALILEO reports small deltas between conditions/models, it may need similarly explicit *paired* statistical tests and multi-benchmark aggregation procedures to back claims.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding a short “significance / reliability” subsection: when comparing two conditions, use paired tests at the item level (McNemar for binary outcomes; extend appropriately for non-binary scores).
- [ ] If we pool multiple tasks/benchmarks, define an explicit aggregation rule (and document how it controls false positives).
- [ ] Add a brief caveat in the evaluation section about temperature-0 nondeterminism from numerical effects; recommend multiple runs or paired testing when feasible.

## Quotes / details to potentially cite

- “We propose a statistically sound hypothesis testing framework based on McNemar’s test allowing to efficiently detect model degradations, while guaranteeing a controlled rate of false positives.” (abstract)
- “The crucial insight is that we have to confront the model scores on each sample, rather than aggregated on the task level.” (abstract)
- “We find that with our tests even empirical accuracy degradations of 0.3% can be confidently attributed to actual degradations rather than noise.” (abstract)
