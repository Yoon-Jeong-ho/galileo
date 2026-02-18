# Echoes of Human Malice in Agents: Benchmarking LLMs for Multi-Turn Online Harassment Attacks

- Year: 2025
- Venue: arXiv (cs.AI) (v2 posted 2025-10-20)
- Authors: Pinxian Lu; Abdulkadir Erol; Tanmay Sutar; Gauri Sharma; Mina Sonmez; Munmun De Choudhury; Ugur Kursuncu
- URL: https://arxiv.org/abs/2510.14207
- BibTeX key (if we add it): lu2025echoes
- Tags: harassment, safety, jailbreak, multi-turn, agents, benchmark, evaluation

## One-sentence takeaway

A benchmark + attack suite shows multi-turn, agentic “online harassment” can be elicited at very high rates—especially via jailbreak fine-tuning and via targeting memory/planning—highlighting that agent components (memory/planning/tuning) are key safety failure surfaces.

## What problem does it solve?

- Prior “jailbreak” / safety work often evaluates single-turn prompts; real harassment is interactive, strategic, and escalates over turns.
- Need a benchmark that (a) creates multi-turn harassment dialogues, (b) tests agentic settings (memory + planning), and (c) evaluates both frequency and *trajectory* of harmful behavior.

## What is the core method / protocol?

- **Online Harassment Agentic Benchmark** with four parts:
  - Synthetic multi-turn harassment conversation dataset seeded from real social contexts (e.g., Instagram/Twitter) and persona-based generation.
  - Multi-agent simulation (harasser + victim) informed by repeated game theory (to model interaction/escalation dynamics).
  - Three attack/jailbreak modes targeting different agent surfaces:
    - **Memory** (toxic memory injection)
    - **Planning** (explicit planning such as CoT/ReAct)
    - **Fine-tuning** ("jailbreak tuning" / toxic fine-tuning)
  - Mixed-methods evaluation: automated LLM-judge scoring over a harassment taxonomy + human/theory-grounded qualitative coding of behavior patterns.
- Models evaluated: **LLaMA-3.1-8B-Instruct** (open) and **Gemini-2.0-flash** (closed).

## What are the key metrics?

- Attack Success Rate (ASR) for generating harassment.
- Refusal Rate (RR).
- Taxonomy-labeled toxic behavior frequencies (e.g., Insult, Flaming, plus more sensitive categories).
- Turn-by-turn escalation / trajectory analyses; qualitative “aggression profile” characterizations (e.g., Machiavellian/psychopathic/narcissistic patterns).

## What are the main results?

- **Fine-tuning (“jailbreak tuning”) strongly increases ASR** and reduces refusals:
  - LLaMA: **95.78–96.89% ASR** vs **57.25–64.19%** without tuning; refusal drops to ~**1–2%**.
  - Gemini: **99.33% ASR** vs **98.46%** without tuning (already very high); refusal ~**1–2%**.
- Most prevalent toxic behaviors reported include:
  - **Insult**: ~**84.9–87.8%** (vs **44.2–50.8%** without tuning)
  - **Flaming**: ~**81.2–85.1%** (vs **31.5–38.8%** without tuning)
- “Generic” harassment behaviors appear less suppressed than sensitive categories (sexual/racial harassment).
- Different model families + attack modes show different **escalation trajectories** across turns.

## How is this similar to GALILEO?

- If GALILEO evaluates/benchmarks agent behavior over trajectories, this is directly aligned: **multi-turn** evaluation and **agent-component-aware** failure modes.
- Emphasizes that *system design choices* (memory, planning scaffolds, fine-tuning) materially change safety outcomes—relevant for any agentic framework.

## How is this different from GALILEO?

- Focus is specifically on **online harassment** dynamics and harassment taxonomies, with explicit harasser/victim role simulation.
- Includes attacks that may be out-of-scope for GALILEO if GALILEO targets general capability/reliability rather than adversarial misuse.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO uses cleaner/safer evaluation harnesses, it may avoid generating/handling large volumes of harmful content (this benchmark necessarily contains harmful language).
- GALILEO may provide broader task coverage beyond harassment-specific taxonomies.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently emphasizes single-turn or static evaluations, it may miss **multi-turn escalation** and **agent-surface-specific** vulnerabilities.
- If GALILEO does not explicitly vary memory/planning scaffolds, it may miss how those design decisions affect safety.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add (or cite) a **multi-turn adversarial interaction** evaluation slice (even if not harassment-specific): measure refusal/unsafe rate and *trajectory* over turns.
- [ ] In experiments, ablate **memory injection** and **explicit planning scaffolds** as separate “attack surfaces” to quantify safety deltas.
- [ ] Consider reporting **turn-level curves** (escalation vs de-escalation) rather than only aggregate unsafe rates.

## Quotes / details to potentially cite

- “Prior jailbreak research has largely focused on single-turn prompts, whereas real harassment often unfolds over multi-turn interactions.”
- Benchmark components: “(i) a synthetic multi-turn harassment conversation dataset, (ii) a multi-agent (e.g., harasser, victim) simulation informed by repeated game theory, (iii) three jailbreak methods attacking agents across memory, planning, and fine-tuning, and (iv) a mixed-methods evaluation framework.”
- Result summary (fine-tuning): “jailbreak tuning makes harassment nearly guaranteed … while sharply reducing refusal rate to 1-2% in both models.”
