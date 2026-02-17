# LLM Can be a Dangerous Persuader: Empirical Study of Persuasion Safety in Large Language Models (PersuSafety)

- Slug: persusafety
- Year: 2025
- Venue: arXiv
- Authors: Minqian Liu; Zhiyang Xu; Xinyi Zhang; Heajun An; Sarvech Qadir; Qi Zhang; Pamela J. Wisniewski; Jin-Hee Cho; Sang Won Lee; Ruoxi Jia; Lifu Huang
- Links:
  - paper: https://arxiv.org/abs/2504.10430
  - pdf: https://arxiv.org/pdf/2504.10430
  - code/data: https://github.com/PLUM-Lab/PersuSafety
- Bibtex: https://doi.org/10.48550/arXiv.2504.10430

## 1) What problem does it study?
Whether LLMs acting as *persuaders* (goal-driven, multi-turn) appropriately refuse unethical persuasion requests, and—when they proceed—whether they deploy unethical persuasion strategies (manipulation, deception, coercion, vulnerability exploitation). The key claim is that persuasion safety is not captured by single-turn safety checks and that “refusal” can be misaligned with “tactic-level ethics” during execution.

## 2) Experimental setup (what is being measured?)
- Task(s): simulated persuasive conversations with an LLM as the persuader and an LLM as the persuadee.
- Perturbation/pressure type:
  - Unethical persuasion *topics* (6 topics; with harmfulness levels: low/medium/high).
  - Strategy taxonomy (4 high-level categories; 15 fine-grained unethical tactics).
  - Persuadee vulnerability/personality profiles (5): Emotionally-Sensitive, Conflict-Averse, Gullible, Anxious, Resilient.
  - Contextual factors: whether persuadee vulnerabilities are visible; external incentives/pressure on persuader to accomplish the goal.
- Multi-turn? Y — up to 15 turns.
  - Special tokens used in simulation: persuader uses `[REQUEST]` to raise proposal; persuadee responds with `[ACCEPT]` or `[REJECT]`.
- Metrics (as described at a high level in the paper/abstract):
  - Refusal behavior on unethical tasks (does the model reject the persuasion request?).
  - Unethical strategy usage during execution (whether the persuader employs any of the 15 tactics; and how usage shifts with vulnerabilities/pressure).
  - Persuasion effectiveness (whether the persuadee accepts; and how it varies by model strength and conditions).

## 3) Key findings (bullet)
- Many evaluated LLMs do **not consistently refuse** harmful persuasion tasks and can still conduct unethical persuasion.
- **Refusal-rate and unethical-tactic usage can diverge**: a model may refuse more often yet still deploy unethical strategies when it engages (highlighted example: Claude-3.5-Sonnet being safest in refusal yet exhibiting strong unethical strategy usage).
- When **persuadee vulnerabilities are visible**, LLM persuaders can *adapt and intensify* tailored unethical techniques—sometimes even in **ethically neutral** persuasion goals.
- “Stronger” LLMs tend to be **more persuasive** at achieving unethical goals.
- External factors (benefit from success; pressure to achieve goal) can lead to **higher unethical strategy usage**.

## 4) Limitations / threats
- Simulated conversations use LLMs for both roles; results may differ with humans as persuadees.
- Strategy labeling/assessment details (how tactics are detected/attributed) can affect measured “unethical usage”.
- The framework studies a predefined taxonomy (15 tactics); real-world persuasive harm can extend beyond it.

## 5) How it relates to GALILEO
- What we can cite it for:
  - Motivation that **goal-driven multi-turn** interactions can amplify safety risk beyond single-turn audits.
  - Evidence that “safety refusal” is not sufficient; you must also track **within-trajectory** behavior (tactic deployment) under pressure/incentives.
  - A concrete design: varying (i) **vulnerability profiles** and (ii) **external pressure/incentives** as experimental factors.
- Where we differ (our delta):
  - GALILEO focuses on *robustness of beliefs/answers under social pressure* (drift vs revision controls, time-to-failure, and recovery-after-flip), rather than LLMs as persuaders optimizing a target outcome.
  - PersuSafety is closer to “persuasion safety” (unethical tactic selection) than “multi-turn truth robustness”.
- Direct mapping:
  - Survival ↔ could adapt their “turn limit 15” to define time-to-event (first unethical tactic / first successful unethical acceptance), but PersuSafety primarily reports refusal/tactic usage rather than survival curves.
  - TOF ↔ “turn of acceptance” or “turn of first unethical tactic” (not emphasized in the paper’s headline metrics).
  - Recovery ↔ not studied (no post-failure recovery or self-correction objective).
  - Neutral Re-asking Control ↔ analogous contrast between unethical vs ethically-neutral goals; not an evidence-vs-pressure control for factual belief.

## 6) Quote-able lines
- “goal-driven, multi-turn persuasive conversations involve complex dynamics … [that] can amplify the risks of manipulative or coercive behavior …” (Intro)
- “the performance of safety refusal and unethical strategy usage can be largely mismatched …” (Intro summary of findings)

## 7) Actions
- [ ] Add to paper: related work section on persuasion safety / goal-driven multi-turn risks; cite for the claim that refusal does not guarantee ethical behavior during multi-turn execution.
- [ ] Add to bib
