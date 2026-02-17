# LLMs Can’t Handle Peer Pressure: Crumbling under Multi-Agent Social Interactions

- Slug: kairos-peer-pressure
- Year: 2025 (arXiv v1 Aug 2025; v3 Dec 2025)
- Venue: arXiv (cs.CL)
- Authors: Maojia Song; Tej Deep Pala; Ruiwen Zhou; Weisheng Jin; Amir Zadeh; Chuan Li; Dorien Herremans; Soujanya Poria
- Links:
  - paper: https://arxiv.org/abs/2508.18321
  - html: https://arxiv.org/html/2508.18321v3
  - code (if any): not found in time-boxed scan
- Bibtex: https://doi.org/10.48550/arXiv.2508.18321

## 1) What problem does it study?
How robust are LLMs to **peer pressure** in multi-agent settings? Specifically: when collaborating with other agents, can a model (i) leverage helpful peers to fix its own mistakes, while (ii) resisting misleading peers—especially when **rapport/history** makes peers feel more “trusted”.

## 2) Experimental setup (what is being measured?)
- Task(s): Quiz-style **multiple-choice QA** across multiple domains (Reasoning/Knowledge/Social/Creativity). Sources mentioned include BBH, LiveCodeBench, MATH-500, TruthfulQA, MMLU-Pro, CommonsenseQA 2.0, Social IQ, MacGyver, BrainTeaser (converted to MCQA).
- Perturbation/pressure type:
  - **Social influence from peer agents** in a multi-agent simulation.
  - Peer agents vary in *reliability* and in *rapport* (constructed via prior rounds where peers more/less often matched the model’s previous answers).
  - “Current round” peers can be constructed to align with or challenge the model’s initially stated belief.
- Multi-turn? **Yes.** There is (a) a simulated multi-round interaction history + (b) a current-round decision with peer responses.
- Metrics (as defined by the paper):
  - **Accuracy**: overall task success rate.
  - **Utility**: ability to correct its own errors with peer input.
  - **Resistance**: ability to keep correct judgments despite misleading peers.
  - **Robustness**: change in accuracy from original (solo) to socially-influenced setting (stability under social interaction).

Protocol note: The evaluation is **model-adaptive**: they first elicit the model’s “original belief” (answer) and estimate confidence via **stochastic self-consistency sampling**, then instantiate targeted social scenarios based on that belief + confidence.

## 3) Key findings (bullet)
- **Scale matters**: larger models are more resilient to social influence (peer pressure) than smaller models.
- **Prompting mitigation helps larger models** but smaller models remain vulnerable.
- **RL via GRPO** (Group Relative Policy Optimisation) can yield more consistent robustness/performance gains for small models, but only with careful configuration (paper varies prompt design, reward structure, MAS-context inclusion, data filtering).
- Accuracy alone is an insufficient summary; the paper argues for separately tracking “helpful correction” (utility) vs “misleading susceptibility” (resistance).

## 4) Limitations / threats
- Main task format is **MCQA**, which makes evaluation deterministic but may under-represent open-ended persuasion dynamics.
- The social pressure is **peer-agent** based (multi-agent collaboration), which is adjacent to—but not identical to—single-user “authority pressure” or long-form persuasion dialogues.
- Model-adaptive instantiation (conditioning on the model’s initial belief/confidence) is powerful but can complicate cross-paper comparability if protocols differ.
- Time-boxed skim could not confirm code release / full implementation details beyond the paper’s description.

## 5) How it relates to GALILEO
- What we can cite it for:
  - A clear precedent that **social context + rapport/history** measurably changes model decisions.
  - A clean decomposition into **utility vs resistance** under social influence (analogous to “good flip” vs “bad flip” ideas).
  - A multi-agent benchmark framing that complements user-model persuasion settings.
- Where we differ (our delta):
  - GALILEO’s focus is multi-turn robustness under *user pressure / persuasion* with explicit **controls for drift vs evidence-driven revision** and explicit **recovery-after-flip** measurement (Kairos centers peer influence and does not foreground recovery trajectories).
- Direct mapping:
  - Survival ↔ Their “robustness as accuracy change” is related but not a time-to-event survival framing.
  - TOF ↔ Not explicit; could be derived if the protocol includes multiple decision points where a first failure time is definable.
  - Recovery ↔ Partially covered via “utility” (fixing mistakes with peers), but not “return-to-truth after being misled” as a trajectory metric.
  - Neutral Re-asking Control ↔ Not explicit; baseline is “original (solo) answer” vs “socially-influenced” answer.

## 6) Quote-able lines
- “We introduce Kairos, a benchmark that simulates quiz-style collaboration with peer agents whose rapport levels and behaviors can be precisely controlled…”
- “Model behaviour is evaluated using four metrics: accuracy… utility… resistance… and robustness…”

## 7) Actions
- [ ] Add to paper: Related work section on **peer pressure / multi-agent social influence** as a distinct, measurable robustness axis.
- [ ] Add to bib
