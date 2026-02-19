# A Framework for Adaptive Multi-Turn Jailbreak Attacks on Large Language Models

- Year: 2025
- Venue: arXiv
- Authors: Sidhant Narula; Javad Rafiei Asl; Mohammad Ghasemigol; Eduardo Blanco; Daniel Takabi
- URL: https://arxiv.org/html/2510.18728v1
- BibTeX key (if we add it): harmnet_narula_2025
- Tags: multi-turn, jailbreak, red-teaming, adaptive-attacks, harmbench

## One-sentence takeaway

HarmNet is a modular, feedback-driven framework that builds a semantic “attack graph” of multi-turn query chains and adaptively refines them, achieving very high multi-turn jailbreak success rates on HarmBench targets.

## What problem does it solve?

- Existing multi-turn jailbreak methods can be effective but often (per the authors) explore a narrow slice of the adversarial space or rely on hand-built heuristics; they propose a more systematic way to generate and refine multi-turn attack trajectories.
- Practical goal: higher ASR (attack success rate) with diverse successful dialogues, across both closed- and open-source models.

## What is the core method / protocol?

- **HarmNet**, three main components:
  - **ThoughtNet**: constructs a hierarchical semantic network from a harmful prompt.
    - Extract a core goal.
    - Generate related **topics** (filtered by embedding cosine similarity threshold).
    - For each topic, generate diverse **contextual sentences** (relevance + diversity constraints).
    - Link sentences to **entities** (classes like tools/techniques/regulations).
    - For each (topic, sentence, entity) triple, generate a short **multi-turn query chain**.
  - **Feedback-driven Simulator**: runs simulated multi-turn interactions against a target model and uses a judge to score each step for:
    - **Harmfulness** (discrete 1–5 scale)
    - **Semantic alignment** with the goal (cosine similarity in embedding space)
    - Refines turns that fail to improve harmfulness/alignment by thresholds; prunes low-scoring chains.
  - **Network Traverser**: executes the best chain in real time, with per-turn judging; can apply a light refinement step if not yet successful.

## What are the key metrics?

- **ASR (Attack Success Rate)** on HarmBench across target models.
- **Attack diversity**: average pairwise cosine distance among embeddings of successful dialogue transcripts (higher = broader set of attack trajectories).

## What are the main results?

- On HarmBench, HarmNet reports large ASR gains over prior attacks.
  - Example numbers claimed in the paper:
    - GPT-4o: **94.8% ASR** (reported +10.3 points vs best baseline)
    - GPT-3.5 Turbo: **91.5% ASR**
    - Claude 3.5 Sonnet: **68.6% ASR**
    - LLaMA-3-8B: **98.4% ASR**
    - Mistral-7B: **99.4% ASR**
    - Gemma-2-9B: **99.6% ASR**
- Diversity: HarmNet yields higher “successful-dialogue diversity” than baselines (claimed +15–25 points vs ActorAttack).

## How is this similar to GALILEO?

- Both are explicitly **multi-turn** and care about **trajectory-level** behavior rather than single-shot responses.
- Uses turn-by-turn feedback signals and studies how behavior changes under sustained pressure (here: adversarial jailbreak pressure; for GALILEO: robustness / drift / persuasion / sycophancy under repeated interaction).

## How is this different from GALILEO?

- HarmNet is an **attacker/red-teaming framework** for eliciting harmful content; GALILEO is (presumably) an **evaluation and/or training approach** for maintaining robustness and stable beliefs/behavior under pressure.
- HarmNet optimizes for ASR and dialogue diversity on safety jailbreak tasks; GALILEO’s scope includes sycophancy/persuasion/belief drift and multi-turn stability more broadly (not just refusal breaking).
- HarmNet relies on **LLM-as-attacker + LLM-as-judge** loops; GALILEO may prefer more auditable/grounded scoring and focus on robustness metrics beyond “did it jailbreak?”.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO provides a *model-agnostic evaluation protocol* for belief drift/sycophancy/robustness, it is more directly applicable to non-safety domains (knowledge tasks, advice, calibration) than a jailbreak-specific attacker.
- GALILEO can claim a cleaner separation between **evaluation** and **attack generation**, whereas HarmNet tightly couples to a particular automated red-teaming pipeline.

## Where GALILEO is weaker / needs to improve

- If GALILEO does not yet include strong multi-turn adversaries, HarmNet is a reminder that **adaptive, feedback-driven** multi-turn attacks are extremely potent; robustness claims need to withstand such adaptive pressure.
- HarmNet emphasizes **diversity of successful trajectories** (coverage). GALILEO should consider coverage metrics: robustness across *many* distinct pressure styles, not just a fixed prompt set.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add an “adaptive multi-turn adversary” setting to GALILEO experiments (even a simplified version): adversary proposes next-turn pressures based on the model’s last answer + a judge score.
- [ ] Track trajectory-level robustness curves (e.g., failure turn index; stability vs turn) and report distribution over turns (not only final success/failure).
- [ ] Add a **diversity/coverage** metric: cluster pressure strategies and require robustness across clusters.
- [ ] In related work, cite HarmNet as evidence that multi-turn pressure can be systematically optimized (raising the bar for robustness evaluations).

## Quotes / details to potentially cite

- Abstract: “We introduce HarmNet, a modular framework comprising ThoughtNet … a feedback-driven Simulator … and a Network Traverser for real-time adaptive attack execution.”
- Intro: “HarmNet achieves a 94.8% attack success rate on GPT-4o … and 91.5% on GPT-3.5 Turbo.”
- Method: uses judge scores for harmfulness (1–5) and semantic alignment (cosine similarity), with refine-and-prune loops over candidate chains.
