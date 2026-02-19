# Lost in Simulation: LLM-Simulated Users are Unreliable Proxies for Human Users in Agentic Evaluations

- Year: 2026
- Venue: arXiv
- Authors: Preethi Seshadri; Samuel Cahyawijaya; Ayomide Odumakinde; Sameer Singh; Seraphina Goldfarb-Tarrant
- URL: https://arxiv.org/abs/2601.17087
- BibTeX key (if we add it): Seshadri2026LostInSimulation
- Tags: agentic-eval, user-simulation, tau-bench, calibration, robustness, fairness

## One-sentence takeaway

LLM-based user simulators (as used in agentic benchmarks like τ-Bench) are **not robust, miscalibrated vs. real users, and uneven across demographics**, risking misleading conclusions about agent capability.

## What problem does it solve?

- Agentic benchmarks often replace humans with an LLM “user” to scale evaluation, but it is unclear whether this is (i) **robust** to the choice of simulator model, (ii) **valid** as a proxy for real user interactions/outcomes, and (iii) **fair** across different user populations/dialects.

## What is the core method / protocol?

- Case study on **τ-Bench retail** tasks.
- **Robustness**: hold the agent fixed (GPT-4o) and vary the **user simulation LLM**, measuring how agent success rates change.
- **Validity + fairness**: run a **human user study** (participants in US, India, Kenya, Nigeria), using a difficulty-balanced subset of τ-Bench retail tasks.
  - US participants include **White SAE** and **Black AAVE** speakers; US also stratified by age.
- Compare simulated-user success rates to human-user success rates across difficulty levels; quantify mismatch via an **ECE-style calibration error** between human and simulated success rates.
- Also analyze qualitative conversational differences (e.g., question-asking/politeness artifacts).

## What are the key metrics?

- **Task success rate** (τ-Bench’s automated success criterion).
- **Human–LLM calibration error** (ECE-style): weighted average absolute difference between success rates with humans vs. with simulated users across discrete difficulty buckets.
- Group-sliced metrics (dialect/country; age strata for US) to assess fairness.

## What are the main results?

- **Non-robustness**: swapping the user-simulator LLM can change the measured agent success rate by **up to ~9 percentage points**.
- **Systematic miscalibration**: simulated-user evaluations **underestimate** agent performance on the hardest tasks and **overestimate** on moderately difficult ones.
- **Fairness gaps**: AAVE speakers see **worse success rates** and **larger calibration errors** than SAE speakers; disparities **compound with age**.
- **Uneven proxy quality**: simulators perform worst (as proxies) for **AAVE** and **Indian English** users.
- **Behavioral artifacts**: simulated users are more “assistant-like” (notably more polite / more question-asking), and expose different failure patterns than human interactions.

## How is this similar to GALILEO?

- Reinforces the high-level thesis that **multi-turn evaluation is fragile** and can be distorted by the evaluation protocol itself (here: the “user” distribution).
- Supports the need for **calibration-aware** and **population-aware** reporting rather than single aggregate success numbers.

## How is this different from GALILEO?

- Focuses on **agentic tool-use benchmark methodology** (τ-Bench) and the validity of **user simulation**, rather than directly proposing a new robustness/steering benchmark.
- Uses a **human user study** + calibration analysis as the core contribution, rather than adversarial prompting / pressure protocols.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO uses **controlled, explicit pressure / evidence conditions**, it may provide a cleaner separation of *legitimate revision* vs. *protocol-induced drift* than τ-Bench-style end-to-end success metrics.
- GALILEO can more directly recommend **diagnostic metrics** (time-to-failure, recovery, etc.) beyond a single success label.

## Where GALILEO is weaker / needs to improve

- If GALILEO relies heavily on **simulated interlocutors**, this paper is a warning that results may be **model-dependent** and **demographically miscalibrated**.
- GALILEO may need stronger guidance on when simulation is acceptable and what **validation** is required.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a short “**simulated user caveat**” paragraph: agentic / interactive eval results can vary materially with the user model; cite this paper.
- [ ] If GALILEO has any interactive benchmark component, run a **robustness slice**: vary the simulated interlocutor model(s) and report variance (min/max or CI).
- [ ] Add a “**calibration vs. humans**” framing where feasible: even small human studies can detect systematic miscalibration across difficulty.
- [ ] If reporting demographic slices, explicitly discuss how simulation may be **non-uniformly valid** across populations.

## Quotes / details to potentially cite

- “Agentic benchmarks increasingly rely on LLM-simulated users … yet the robustness, validity, and fairness of this approach remain unexamined.”
- Reported finding: measured success rates can vary by **up to 9 percentage points** across different user LLMs.
- Simulated users introduce artifacts such as increased **question-asking and politeness** vs. humans.
