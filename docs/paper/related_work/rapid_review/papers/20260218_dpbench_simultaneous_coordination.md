# DPBench: Large Language Models Struggle with Simultaneous Coordination

- Year: 2026
- Venue: arXiv preprint
- Authors: Prashanth Busireddygari et al.
- URL: https://arxiv.org/abs/2602.13255
- BibTeX key (if we add it): dpbench2026
- Tags: multi-agent, coordination, benchmark, concurrency, deadlock

## One-sentence takeaway

DPBench shows frontier LLM agents can coordinate well in turn-based/sequential settings but often deadlock under simultaneous resource contention due to convergent (identical) strategies, suggesting the need for explicit coordination mechanisms.

## What problem does it solve?

- We lack targeted benchmarks for **simultaneous** (not turn-based) multi-agent coordination under shared resource contention.
- Existing multi-agent LLM evals are often sequential, which can mask failures that appear when agents must decide at the same time.

## What is the core method / protocol?

- Introduces **DPBench**, a benchmark instantiating the classic **Dining Philosophers** coordination problem for LLM agents.
- Evaluates across **8 conditions** varying:
  - decision timing: sequential vs simultaneous
  - group size: 3 vs 5 agents
  - communication: enabled vs disabled
- Defines standardized outcome metrics (paper mentions deadlock rate, throughput, fairness among others).
- Key analysis claim: failures stem from **convergent reasoning**—agents independently pick the same “reasonable” local policy which, when executed simultaneously, guarantees circular wait / deadlock.

## What are the key metrics?

- Deadlock rate
- Throughput (successful progress/completions)
- Fairness (how evenly agents get to “eat” / succeed)
- (Paper states 6 standardized metrics total.)

## What are the main results?

- Strong asymmetry:
  - Sequential mode: near-0% deadlock (best model reported: GPT-5.2 at 0% in sequential).
  - Simultaneous mode: deadlocks can be very high (reported up to >95% in some conditions; GPT-5.2 reported 25–95% deadlock depending on condition).
- Communication does **not** reliably fix coordination; can sometimes **increase** deadlock.

## How is this similar to GALILEO?

- Directly relevant to evaluating **multi-agent reliability** and failure modes when coordination is required.
- Highlights that “works in dialogue/turn-taking” does not imply “works under concurrency,” which is likely relevant for any GALILEO setting with shared tools/resources or parallel agent actions.

## How is this different from GALILEO?

- DPBench is a **micro-benchmark** focused on a stylized concurrency deadlock scenario (Dining Philosophers), not an end-to-end task benchmark.
- Primary emphasis is on **simultaneity** and resource contention; GALILEO may target broader collaboration, planning, or tool use.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO includes more realistic tasks, it may better reflect practical agent workloads (beyond a single coordination motif).

## Where GALILEO is weaker / needs to improve

- If GALILEO evaluations are mostly sequential/turn-based, it may be under-testing simultaneous coordination failures that DPBench surfaces.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add at least one GALILEO eval condition where agents must commit actions **simultaneously** (or with limited observation) and contend for a shared resource.
- [ ] Track deadlock-like outcomes explicitly (timeouts, circular waits, “everyone chose the same lock ordering,” etc.).
- [ ] Consider testing external coordination mechanisms (token/lock server, randomized backoff, leader election) vs pure LLM emergent coordination.

## Quotes / details to potentially cite

- “LLMs coordinate effectively in sequential settings but fail when decisions must be made simultaneously, with deadlock rates exceeding 95% under some conditions.” (abstract)
- Failure attributed to “convergent reasoning … identical strategies that, when executed simultaneously, guarantee deadlock.” (abstract)
- “Enabling communication does not resolve this problem and can even increase deadlock rates.” (abstract)
