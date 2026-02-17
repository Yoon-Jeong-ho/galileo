# PatientHub: A Unified Framework for Patient Simulation

- Year: 2026
- Venue: arXiv (Work in progress)
- Authors: Sahand Sabour; TszYam Ng; Minlie Huang
- URL: https://arxiv.org/abs/2602.11684
- BibTeX key (if we add it): patienthub2026sabour
- Tags: multi-turn, patient-simulation, framework, reproducibility, evaluation, mental-health

## One-sentence takeaway

PatientHub proposes a modular, standardized pipeline for specifying and running LLM-based simulated patients, aiming to make patient-simulation methods comparable and reproducible.

## What problem does it solve?

- Patient-simulation work for counselor training / therapeutic assessment is fragmented across incompatible persona formats, prompts, orchestration setups, and evaluation metrics.
- This fragmentation makes it hard to reproduce results, fairly compare methods, or plug in new metrics.

## What is the core method / protocol?

- A **unified framework** that standardizes:
  - how a simulated patient is *defined* (profile/spec)
  - how simulators are *composed* and *deployed* in multi-turn dialogue
  - how runs are *evaluated* with a common interface
- Provides reference implementations of multiple representative simulation methods as **case studies**.
- Demonstrates extensibility by adding **two new simulator variants** to show low overhead for new method prototyping.

## What are the key metrics?

- Not a single new metric; the contribution is infrastructure to support **standardized cross-method evaluation** and **custom metric integration**.
- Emphasis is on reproducible pipelines and apples-to-apples comparisons (metric definitions are intended to be swappable).

## What are the main results?

- Qualitative/engineering result: multiple existing patient-simulation approaches can be implemented under a shared abstraction and evaluated in a standardized way.
- Claims: reduces infrastructure overhead, lowers barrier to new method development, and enables cross-method/cross-model benchmarking.

## How is this similar to GALILEO?

- Same meta-goal: make **multi-turn evaluation** more systematic and comparable (standardized protocols, clearer reporting).
- Highlights that, for multi-turn systems, *pipeline design* (data formats + orchestration + metrics) strongly affects reproducibility.

## How is this different from GALILEO?

- Domain focus is **patient-centered / counseling-style dialogue** and patient simulation (role-play), not social-pressure belief drift per se.
- Contribution is primarily a **framework/software standardization**; GALILEO’s core is a **measurement protocol + analysis of multi-turn robustness under pressure**.

## Where GALILEO is stronger / cleaner (if true)

- GALILEO can contribute sharper *construct validity* around pressure-driven drift vs evidence-driven revision, time-to-failure, and recovery dynamics.
- GALILEO’s metrics/protocols are likely more directly transferable to general LLM robustness discussions outside a single application domain.

## Where GALILEO is weaker / needs to improve

- If we want broader impact in applied safety/health settings, we may need cleaner guidance on **how to package protocols/metrics** so others can adopt them as easily as a framework like PatientHub.

## Action items for GALILEO (experiments / method / writing)

- [ ] In Related Work, cite PatientHub as evidence that **standardization/reproducibility is a recognized bottleneck** in multi-turn dialogue evaluation (applied domain).
- [ ] Consider whether GALILEO artifacts (data schema + run harness + metric hooks) should be described as a “drop-in evaluation harness”, even if not positioned as a general framework.

## Quotes / details to potentially cite

- “Prior work is fragmented: existing approaches rely on incompatible, non-standardized data formats, prompts, and evaluation metrics, hindering reproducibility and fair comparison.”
- “PatientHub … standardizes the definition, composition, and deployment of simulated patients.”
- Code: https://github.com/Sahandfer/PatientHub
